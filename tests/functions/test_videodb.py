from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import pixeltable as pxt

from ..utils import (
    get_video_files,
    rerun_on_network_error,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


@pytest.fixture
def mock_connection(init_env: None, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    skip_test_if_not_installed('videodb')
    import videodb  # type: ignore[import-untyped]

    outputs = {
        'transcript': [
            {'start': 0, 'end': 1.5, 'data': {'text': 'Hello there.'}},
            {'start': 1.5, 'end': 2, 'data': {'text': ' '}},
            {'start': 2, 'end': 3, 'data': {'text': 'General Kenobi.'}},
        ],
        'scene': [{'start': 0, 'end': 10, 'data': {'text': 'A cooking show.'}}],
        'ocr': [{'start': 0, 'end': 5, 'data': {'text': 'EXIT'}}, {'start': 5, 'end': 10, 'data': {'text': ''}}],
        'brands': [
            {
                'start': 0,
                'end': 5,
                'data': {'brand_names': ['Brinks'], 'mentions': [{'brand_name': 'Brinks', 'reason': 'Logo on truck.'}]},
            },
            {'start': 5, 'end': 10, 'data': {'brand_names': [], 'mentions': []}},
        ],
        'activity': [{'start': 0, 'end': 10, 'data': {'activity': 'cooking'}}],
        'location': [{'start': 0, 'end': 10, 'data': {'location': 'kitchen'}}],
        'objects': [
            {
                'start': 0,
                'end': 10,
                'data': {
                    'frames': [
                        {'timestamp': 0.0, 'detections': []},
                        {
                            'timestamp': 1.0,
                            'detections': [
                                {
                                    'label': 'cup',
                                    'score': 0.9,
                                    'box': {'box': [0.1, 0.2, 0.3, 0.4], 'unit': 'normalized'},
                                }
                            ],
                        },
                    ]
                },
            }
        ],
    }

    def understand(analyzers: list, segmentation: dict | None = None, transform: dict | None = None) -> MagicMock:
        understanding = MagicMock()
        understanding.get_analyzer.return_value.is_successful = True
        understanding.get_analyzer.return_value.get_output.return_value = {'scenes': outputs[analyzers[-1]['name']]}
        return understanding

    video = MagicMock(id='m-test')
    video.understand.side_effect = understand
    collection = MagicMock()
    collection.upload.return_value = video
    collection.get_video.return_value = video
    conn = MagicMock()
    conn.get_collection.return_value = collection
    monkeypatch.setenv('VIDEO_DB_API_KEY', 'test-key')
    monkeypatch.setattr(videodb, 'connect', MagicMock(return_value=conn))
    return conn


class TestVideoDBLocal:
    def test_udfs(self, uses_db: None, mock_connection: MagicMock) -> None:
        import videodb

        from pixeltable.functions.videodb import describe_scenes, transcribe, upload

        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(videodb_id=upload(t.video, collection_id='c-test'))
        t.add_computed_column(transcript=transcribe(t.videodb_id, collection_id='c-test', language='en'))
        t.add_computed_column(
            scenes=describe_scenes(
                t.videodb_id,
                collection_id='c-test',
                prompt='Describe it.',
                segmentation={'type': 'time', 'seconds': 10},
            )
        )
        video_file = get_video_files()[0]
        validate_update_status(t.insert(video=video_file), 1)

        res = t.collect()
        assert res['videodb_id'][0] == 'm-test'
        assert res['transcript'][0] == [
            {'start': 0.0, 'end': 1.5, 'text': 'Hello there.'},
            {'start': 2.0, 'end': 3.0, 'text': 'General Kenobi.'},
        ]
        assert res['scenes'][0] == [{'start': 0.0, 'end': 10.0, 'description': 'A cooking show.', 'fields': {}}]

        videodb.connect.assert_called_with(api_key='test-key')
        mock_connection.get_collection.assert_called_with('c-test')
        collection = mock_connection.get_collection.return_value
        collection.upload.assert_called_once_with(file_path=video_file)
        collection.get_video.assert_called_with('m-test')
        video = collection.get_video.return_value
        assert [c.kwargs for c in video.understand.call_args_list] == [
            {
                'analyzers': [{'type': 'spoken_words', 'name': 'transcript', 'config': {'language': 'en'}}],
                'segmentation': None,
                'transform': None,
            },
            {
                'analyzers': [{'type': 'vlm', 'name': 'scene', 'config': {'prompt': 'Describe it.'}}],
                'segmentation': {'type': 'time', 'seconds': 10},
                'transform': None,
            },
        ]
        # every call closes the connection it opened
        assert mock_connection.session.close.call_count == mock_connection.get_collection.call_count == 3

    def test_scene_options(self, uses_db: None, mock_connection: MagicMock) -> None:
        from pixeltable.functions.videodb import describe_scenes

        video = mock_connection.get_collection.return_value.get_video.return_value
        video.understand.side_effect = None
        video.understand.return_value.get_analyzer.return_value.get_output.return_value = {
            'scenes': [{'start': 0, 'end': 10, 'data': {'summary': 'Cooking.', 'people_count': 2}}]
        }
        t = pxt.create_table('test_tbl', {'videodb_id': pxt.String})
        t.add_computed_column(
            scenes=describe_scenes(
                t.videodb_id,
                prompt='Summarize. Said: {{inputs.transcript}} Shown: {{inputs.signs}}',
                schema={'summary': 'string', 'people_count': 'integer'},
                inputs=['spoken_words', {'type': 'ocr', 'name': 'signs'}],
                model='mini',
                sampling={'strategy': 'uniform', 'frame_count': 4},
                segmentation={'type': 'time', 'seconds': 10},
                transform={'resolution': '480p'},
            )
        )
        validate_update_status(t.insert(videodb_id='m-test'), 1)

        assert t.collect()['scenes'][0] == [
            {'start': 0.0, 'end': 10.0, 'description': '', 'fields': {'summary': 'Cooking.', 'people_count': 2}}
        ]
        assert video.understand.call_args.kwargs == {
            'analyzers': [
                {'type': 'spoken_words', 'name': 'transcript'},
                {'type': 'ocr', 'name': 'signs'},
                {
                    'type': 'vlm',
                    'name': 'scene',
                    'inputs': ['transcript', 'signs'],
                    'sampling': {'strategy': 'uniform', 'frame_count': 4},
                    'config': {
                        'prompt': 'Summarize. Said: {{inputs.transcript}} Shown: {{inputs.signs}}',
                        'schema': {'summary': 'string', 'people_count': 'integer'},
                        'model': 'mini',
                    },
                },
            ],
            'segmentation': {'type': 'time', 'seconds': 10},
            'transform': {'resolution': '480p'},
        }
        video.understand.return_value.get_analyzer.assert_called_with('scene', refresh=True)

    def test_unsupported_input(self, uses_db: None, mock_connection: MagicMock) -> None:
        from pixeltable.functions.videodb import describe_scenes

        t = pxt.create_table('test_tbl', {'videodb_id': pxt.String})
        t.add_computed_column(scenes=describe_scenes(t.videodb_id, inputs=['faces']))
        t.insert(videodb_id='m-test', on_error='ignore')
        assert 'Unsupported VideoDB input analyzer' in t.select(e=t.scenes.errormsg).collect()['e'][0]

    def test_visual_analyzers(self, uses_db: None, mock_connection: MagicMock) -> None:
        from pixeltable.functions.videodb import (
            detect_brands,
            detect_location,
            detect_objects,
            read_text,
            recognize_activity,
        )

        options = {'model': 'mini', 'sampling': {'strategy': 'uniform', 'frame_count': 2}}
        t = pxt.create_table('test_tbl', {'videodb_id': pxt.String})
        t.add_computed_column(text=read_text(t.videodb_id, transform={'resolution': '480p'}, **options))
        t.add_computed_column(brands=detect_brands(t.videodb_id, **options))
        t.add_computed_column(activity=recognize_activity(t.videodb_id, **options))
        t.add_computed_column(location=detect_location(t.videodb_id, segmentation={'type': 'time', 'seconds': 5}))
        t.add_computed_column(
            objects=detect_objects(
                t.videodb_id,
                model='rtdetr-v2-r50vd',
                sandbox_id='sb-test',
                sampling={'strategy': 'interval', 'every': 1},
            )
        )
        validate_update_status(t.insert(videodb_id='m-test'), 1)

        res = t.collect()
        assert res['text'][0] == [{'start': 0.0, 'end': 5.0, 'text': 'EXIT'}]
        assert res['brands'][0] == [
            {
                'start': 0.0,
                'end': 5.0,
                'brand_names': ['Brinks'],
                'mentions': [{'brand_name': 'Brinks', 'reason': 'Logo on truck.'}],
            }
        ]
        assert res['activity'][0] == [{'start': 0.0, 'end': 10.0, 'activity': 'cooking'}]
        assert res['location'][0] == [{'start': 0.0, 'end': 10.0, 'location': 'kitchen'}]
        assert res['objects'][0] == [
            {
                'timestamp': 1.0,
                'detections': [{'label': 'cup', 'score': 0.9, 'box': [0.1, 0.2, 0.3, 0.4], 'unit': 'normalized'}],
            }
        ]

        video = mock_connection.get_collection.return_value.get_video.return_value
        # The UDFs run concurrently, so the calls are compared in a fixed order
        calls = sorted((c.kwargs for c in video.understand.call_args_list), key=lambda c: c['analyzers'][0]['type'])
        sampling = options['sampling']
        assert calls == [
            {
                'analyzers': [
                    {
                        'type': 'activity_recognition',
                        'name': 'activity',
                        'sampling': sampling,
                        'config': {'model': 'mini'},
                    }
                ],
                'segmentation': None,
                'transform': None,
            },
            {
                'analyzers': [
                    {'type': 'brand_detection', 'name': 'brands', 'sampling': sampling, 'config': {'model': 'mini'}}
                ],
                'segmentation': None,
                'transform': None,
            },
            {
                'analyzers': [{'type': 'location_detection', 'name': 'location'}],
                'segmentation': {'type': 'time', 'seconds': 5},
                'transform': None,
            },
            {
                'analyzers': [
                    {
                        'type': 'object_detection',
                        'name': 'objects',
                        'sampling': {'strategy': 'interval', 'every': 1},
                        'config': {'model': 'rtdetr-v2-r50vd', 'sandbox_id': 'sb-test'},
                    }
                ],
                'segmentation': None,
                'transform': None,
            },
            {
                'analyzers': [{'type': 'ocr', 'name': 'ocr', 'sampling': sampling, 'config': {'model': 'mini'}}],
                'segmentation': None,
                'transform': {'resolution': '480p'},
            },
        ]

    def test_upload_without_video_id(self, uses_db: None, mock_connection: MagicMock) -> None:
        from pixeltable.functions.videodb import upload

        mock_connection.get_collection.return_value.upload.return_value = SimpleNamespace(id='a-test')
        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(videodb_id=upload(t.video))
        t.insert(video=get_video_files()[0], on_error='ignore')
        res = t.select(t.videodb_id, error=t.videodb_id.errormsg).collect()
        assert res['videodb_id'][0] is None
        assert 'did not return a video ID' in res['error'][0]


@pytest.mark.expensive
@pytest.mark.skip_merge_queue
@rerun_on_network_error()
class TestVideoDB:
    def test_transcribe_and_describe_scenes(self, uses_db: None) -> None:
        skip_test_if_not_installed('videodb')
        skip_test_if_no_client('video_db')
        from pixeltable.functions.videodb import describe_scenes, transcribe, upload

        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(videodb_id=upload(t.video))
        t.add_computed_column(transcript=transcribe(t.videodb_id))
        t.add_computed_column(scenes=describe_scenes(t.videodb_id, segmentation={'type': 'time', 'seconds': 10}))
        validate_update_status(t.insert(video=get_video_files(extension='.webm')[0]), 1)
        res = t.collect()
        assert res['videodb_id'][0].startswith('m-')
        assert isinstance(res['transcript'][0], list)
        scenes = res['scenes'][0]
        assert len(scenes) > 0
        assert all(len(scene['description']) > 0 for scene in scenes)

    def test_visual_analyzers(self, uses_db: None) -> None:
        skip_test_if_not_installed('videodb')
        skip_test_if_no_client('video_db')
        from pixeltable.functions.videodb import (
            describe_scenes,
            detect_brands,
            detect_location,
            read_text,
            recognize_activity,
            upload,
        )

        options = {'model': 'mini', 'segmentation': {'type': 'time', 'seconds': 10}}
        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(videodb_id=upload(t.video))
        t.add_computed_column(text=read_text(t.videodb_id, **options))
        t.add_computed_column(brands=detect_brands(t.videodb_id, **options))
        t.add_computed_column(activity=recognize_activity(t.videodb_id, **options))
        t.add_computed_column(location=detect_location(t.videodb_id, transform={'resolution': '480p'}, **options))
        t.add_computed_column(
            scenes=describe_scenes(
                t.videodb_id,
                prompt='Summarize the scene. Visible text: {{inputs.ocr}}',
                schema={'summary': 'string'},
                inputs=['ocr'],
                sampling={'strategy': 'uniform', 'frame_count': 2},
                **options,
            )
        )
        validate_update_status(t.insert(video=get_video_files(extension='.webm')[0]), 1)
        res = t.collect()
        assert isinstance(res['text'][0], list)
        assert isinstance(res['brands'][0], list)
        assert all(len(scene['activity']) > 0 for scene in res['activity'][0])
        assert len(res['location'][0]) > 0
        assert all(len(scene['location']) > 0 for scene in res['location'][0])
        scenes = res['scenes'][0]
        assert len(scenes) > 0
        assert all(len(scene['fields']['summary']) > 0 for scene in scenes)
