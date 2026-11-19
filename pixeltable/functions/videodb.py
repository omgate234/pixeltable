"""
Pixeltable UDFs
that wrap various endpoints from the VideoDB API. In order to use them, you must
first `pip install videodb` and configure your VideoDB API key, by setting the `VIDEO_DB_API_KEY` environment
variable or `video_db.api_key` in the Pixeltable config file.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, Iterator, TypedDict

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.runtime import get_runtime

if TYPE_CHECKING:
    from videodb.collection import Collection  # type: ignore[import-untyped]


class _VideoDBClient:
    """Opens a new videodb Connection for each call: a Connection keeps per-request polling state, so concurrent
    calls can't share one."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @contextlib.contextmanager
    def collection(self, collection_id: str) -> Iterator['Collection']:
        import videodb  # type: ignore[import-untyped]

        conn = videodb.connect(api_key=self.api_key)
        try:
            yield conn.get_collection(collection_id)
        finally:
            conn.session.close()


@register_client('video_db', credential_param='api_key')
def _(api_key: str) -> _VideoDBClient:
    return _VideoDBClient(api_key)


def _videodb_client() -> _VideoDBClient:
    return get_runtime().get_client('video_db')


class TranscriptSegment(TypedDict):
    """One timestamped segment returned by `transcribe()`."""

    start: float
    """Start of the segment in seconds."""
    end: float
    """End of the segment in seconds."""
    text: str


class SceneDescription(TypedDict):
    """One scene returned by `describe_scenes()`."""

    start: float
    """Start of the scene in seconds."""
    end: float
    """End of the scene in seconds."""
    description: str
    """The scene description. Empty when a `schema` is given."""
    fields: dict[str, Any]
    """The fields extracted for the `schema`, if one is given."""


class OnScreenText(TypedDict):
    """The text visible in one scene, as returned by `read_text()`."""

    start: float
    """Start of the scene in seconds."""
    end: float
    """End of the scene in seconds."""
    text: str


class BrandMention(TypedDict):
    """One brand found by `detect_brands()`."""

    brand_name: str
    reason: str
    """Why the brand was detected, such as the logo or text that shows it."""


class SceneBrands(TypedDict):
    """The brands visible in one scene, as returned by `detect_brands()`."""

    start: float
    """Start of the scene in seconds."""
    end: float
    """End of the scene in seconds."""
    brand_names: list[str]
    mentions: list[BrandMention]


class SceneActivity(TypedDict):
    """The main activity of one scene, as returned by `recognize_activity()`."""

    start: float
    """Start of the scene in seconds."""
    end: float
    """End of the scene in seconds."""
    activity: str


class SceneLocation(TypedDict):
    """The setting of one scene, as returned by `detect_location()`."""

    start: float
    """Start of the scene in seconds."""
    end: float
    """End of the scene in seconds."""
    location: str


class ObjectDetection(TypedDict):
    """One object found in a frame by `detect_objects()`."""

    label: str
    """The COCO label of the object, such as `person` or `car`."""
    score: float
    box: list[float]
    """The bounding box as `[x1, y1, x2, y2]`."""
    unit: str
    """The unit of `box`, such as `normalized`."""


class FrameObjects(TypedDict):
    """The objects found in one sampled frame, as returned by `detect_objects()`."""

    timestamp: float
    """Time of the frame in seconds."""
    detections: list[ObjectDetection]


@pxt.udf(is_deterministic=False, resource_pool='request-rate:video_db')
async def upload(video: pxt.Video, *, collection_id: str = 'default') -> str:
    """
    Uploads a video to a VideoDB collection and returns its VideoDB video ID.

    The other UDFs in this module take that ID, so each video is uploaded once. Equivalent to the VideoDB
    `Collection.upload()` method:
    <https://docs.videodb.io/api-reference/collections/upload_to_collection>

    Request throttling:
    Applies the rate limit set in the config (section `video_db`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video: The video to upload.
        collection_id: The VideoDB collection to upload to.

    Returns:
        The VideoDB video ID, such as `m-...`.

    Examples:
        Add a computed column with the VideoDB ID of each video in the column `tbl.video`:

        >>> tbl.add_computed_column(videodb_id=upload(tbl.video))
    """
    Env.get().require_package('videodb')
    client = _videodb_client()

    def run() -> str:
        with client.collection(collection_id) as coll:
            media = coll.upload(file_path=video)
            if media is None or not media.id.startswith('m-'):
                raise pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                    'VideoDB did not return a video ID for the upload.',
                    provider='videodb',
                )
            return media.id

    return await asyncio.to_thread(run)


# Maps each analyzer type to the name of the artifact it produces
_ANALYZER_NAMES = {
    'spoken_words': 'transcript',
    'vlm': 'scene',
    'ocr': 'ocr',
    'brand_detection': 'brands',
    'activity_recognition': 'activity',
    'location_detection': 'location',
    'object_detection': 'objects',
}


def _analyzer(
    type_: str,
    *,
    config: dict[str, Any],
    sampling: dict[str, Any] | None = None,
    inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    analyzer: dict[str, Any] = {'type': type_, 'name': _ANALYZER_NAMES[type_]}
    if inputs:
        analyzer['inputs'] = [a['name'] for a in inputs]
    if sampling is not None:
        analyzer['sampling'] = sampling
    config = {k: v for k, v in config.items() if v is not None}
    if config:
        analyzer['config'] = config
    return analyzer


def _parse_inputs(inputs: list[Any] | None) -> list[dict[str, Any]]:
    # VideoDB resolves `inputs` by analyzer name, so every input needs one
    analyzers = []
    for entry in inputs or []:
        analyzer = {'type': entry} if isinstance(entry, str) else dict(entry)
        if analyzer.get('type') not in _ANALYZER_NAMES:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_ARGUMENT,
                f'Unsupported VideoDB input analyzer {entry!r}; expected one of {list(_ANALYZER_NAMES)}.',
            )
        analyzer.setdefault('name', _ANALYZER_NAMES[analyzer['type']])
        analyzers.append(analyzer)
    return analyzers


def _understand(
    video_id: str,
    collection_id: str,
    analyzers: list[dict[str, Any]],
    segmentation: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Runs VideoDB understanding analyzers on a video and returns the timestamped scenes of the last one."""
    name = analyzers[-1]['name']
    client = _videodb_client()
    with client.collection(collection_id) as coll:
        video = coll.get_video(video_id)
        understanding = video.understand(analyzers=analyzers, segmentation=segmentation, transform=transform)
        understanding.wait_until_complete()
        result = understanding.get_analyzer(name, refresh=True)
        if not result.is_successful:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_ERROR,
                f'VideoDB analyzer {name!r} finished with status {result.status!r} for video {video_id}.',
                provider='videodb',
            )
        output = result.get_output() or {}
        return output.get('scenes') or []


def _scenes_with_data(scenes: list[dict[str, Any]]) -> Iterator[tuple[float, float, dict[str, Any]]]:
    # Skips scenes without output, such as scenes with no visible text
    for scene in scenes:
        data = scene.get('data') or {}
        if any(v not in (None, '', [], {}) for v in data.values()):
            yield float(scene['start']), float(scene['end']), data


@pxt.udf(is_deterministic=False, resource_pool='request-rate:video_db')
async def transcribe(
    video_id: str, *, collection_id: str = 'default', language: str | None = None
) -> list[TranscriptSegment]:
    """
    Transcribes the speech in a VideoDB video and returns the timestamped transcript.

    Runs the VideoDB `spoken_words` analyzer through `Video.understand()`:
    <https://docs.videodb.io/pages/understand/indexing-pipelines/understanding-artifacts>

    Segments without speech are left out, so a silent video returns an empty list.

    Request throttling:
    Applies the rate limit set in the config (section `video_db`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video_id: The VideoDB video ID, as returned by `upload()`.
        collection_id: The VideoDB collection that contains the video.
        language: The spoken language as a code such as `en` or `fr`. If not specified, VideoDB uses its default.

    Returns:
        A list of transcript segments.

    Examples:
        Add a computed column with the transcript of each uploaded video:

        >>> tbl.add_computed_column(videodb_id=upload(tbl.video))
        ... tbl.add_computed_column(transcript=transcribe(tbl.videodb_id))
    """
    Env.get().require_package('videodb')
    analyzer = _analyzer('spoken_words', config={'language': language})
    scenes = await asyncio.to_thread(_understand, video_id, collection_id, [analyzer])
    return [
        TranscriptSegment(start=start, end=end, text=data['text'])
        for start, end, data in _scenes_with_data(scenes)
        if str(data.get('text') or '').strip()
    ]


@pxt.udf(is_deterministic=False, resource_pool='request-rate:video_db')
async def describe_scenes(
    video_id: str,
    *,
    collection_id: str = 'default',
    prompt: str | None = None,
    schema: dict[str, Any] | None = None,
    inputs: list[Any] | None = None,
    model: str | None = None,
    sampling: dict[str, Any] | None = None,
    segmentation: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
) -> list[SceneDescription]:
    """
    Splits a VideoDB video into scenes and describes each scene with a vision-language model.

    Runs the VideoDB `vlm` analyzer through `Video.understand()`:
    <https://docs.videodb.io/pages/understand/indexing-pipelines/understanding-artifacts>

    Each call creates a new understanding run on the video.

    Request throttling:
    Applies the rate limit set in the config (section `video_db`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video_id: The VideoDB video ID, as returned by `upload()`.
        collection_id: The VideoDB collection that contains the video.
        prompt: Instructions for the scene descriptions. It can reference the outputs of `inputs` analyzers with
            `{{inputs.<name>}}` placeholders, such as `{{inputs.transcript}}`. If not specified, VideoDB uses its
            default prompt.
        schema: The fields to extract for each scene, such as `{'activity': 'string', 'people_count': 'integer'}`.
            They are returned in the `fields` of each scene instead of a `description`.
        inputs: Other analyzers whose outputs the model receives as context for each scene, such as
            `['spoken_words', 'ocr']`. Each entry is an analyzer type, or a VideoDB analyzer dict such as
            `{'type': 'ocr', 'name': 'signs'}`. Default names are `transcript`, `ocr`, `brands`, `activity`,
            `location` and `objects`.
        model: The VideoDB model tier, such as `mini`, `basic`, `pro` or `ultra`. If not specified, VideoDB uses
            its default.
        sampling: Which frames the model sees in each scene, such as `{'strategy': 'uniform', 'frame_count': 4}`
            or `{'strategy': 'interval', 'every': 1}`.
        segmentation: How to split the video into scenes, such as `{'type': 'time', 'seconds': 10}` or
            `{'type': 'shot', 'threshold': 30}`. If not specified, VideoDB uses its default.
        transform: Preprocessing before analysis, such as `{'resolution': '480p'}` to reduce cost.

    Returns:
        A list of scene descriptions.

    Examples:
        Add a computed column with scene descriptions every 10 seconds of each uploaded video:

        >>> tbl.add_computed_column(
        ...     scenes=describe_scenes(
        ...         tbl.videodb_id,
        ...         prompt='Describe the people and objects in the scene.',
        ...         segmentation={'type': 'time', 'seconds': 10},
        ...     )
        ... )

        Extract structured fields for each scene, using what was said as context:

        >>> tbl.add_computed_column(
        ...     scenes=describe_scenes(
        ...         tbl.videodb_id,
        ...         prompt='What is happening? What was said: {{inputs.transcript}}',
        ...         schema={'summary': 'string', 'people_count': 'integer'},
        ...         inputs=['spoken_words'],
        ...     )
        ... )
    """
    Env.get().require_package('videodb')
    analyzers = _parse_inputs(inputs)
    analyzers.append(
        _analyzer(
            'vlm', config={'prompt': prompt, 'schema': schema, 'model': model}, sampling=sampling, inputs=analyzers
        )
    )
    scenes = await asyncio.to_thread(_understand, video_id, collection_id, analyzers, segmentation, transform)
    results = []
    for start, end, data in _scenes_with_data(scenes):
        fields = {k: v for k, v in data.items() if k != 'text'}
        results.append(SceneDescription(start=start, end=end, description=data.get('text') or '', fields=fields))
    return results


async def _run_visual(
    type_: str,
    video_id: str,
    collection_id: str,
    model: str | None,
    sampling: dict[str, Any] | None,
    segmentation: dict[str, Any] | None,
    transform: dict[str, Any] | None,
) -> Iterator[tuple[float, float, dict[str, Any]]]:
    Env.get().require_package('videodb')
    analyzer = _analyzer(type_, config={'model': model}, sampling=sampling)
    scenes = await asyncio.to_thread(_understand, video_id, collection_id, [analyzer], segmentation, transform)
    return _scenes_with_data(scenes)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:video_db')
async def read_text(
    video_id: str,
    *,
    collection_id: str = 'default',
    model: str | None = None,
    sampling: dict[str, Any] | None = None,
    segmentation: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
) -> list[OnScreenText]:
    """
    Reads the text visible in each scene of a VideoDB video, such as signs, slides, captions and UI text.

    Runs the VideoDB `ocr` analyzer through `Video.understand()`:
    <https://docs.videodb.io/pages/understand/indexing-pipelines/understanding-artifacts>

    Scenes without visible text are left out.

    Request throttling:
    Applies the rate limit set in the config (section `video_db`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video_id: The VideoDB video ID, as returned by `upload()`.
        collection_id: The VideoDB collection that contains the video.
        model: The VideoDB model tier, such as `mini`, `basic`, `pro` or `ultra`. If not specified, VideoDB uses
            its default.
        sampling: Which frames the model sees in each scene, such as `{'strategy': 'uniform', 'frame_count': 4}`.
        segmentation: How to split the video into scenes, such as `{'type': 'time', 'seconds': 10}` or
            `{'type': 'shot', 'threshold': 30}`. If not specified, VideoDB uses its default.
        transform: Preprocessing before analysis, such as `{'resolution': '480p'}` to reduce cost.

    Returns:
        A list of scenes with their visible text.

    Examples:
        >>> tbl.add_computed_column(on_screen_text=read_text(tbl.videodb_id))
    """
    rows = await _run_visual('ocr', video_id, collection_id, model, sampling, segmentation, transform)
    return [OnScreenText(start=start, end=end, text=data.get('text') or '') for start, end, data in rows]


@pxt.udf(is_deterministic=False, resource_pool='request-rate:video_db')
async def detect_brands(
    video_id: str,
    *,
    collection_id: str = 'default',
    model: str | None = None,
    sampling: dict[str, Any] | None = None,
    segmentation: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
) -> list[SceneBrands]:
    """
    Finds the logos, brand names and product labels visible in each scene of a VideoDB video.

    Runs the VideoDB `brand_detection` analyzer through `Video.understand()`:
    <https://docs.videodb.io/pages/understand/indexing-pipelines/understanding-artifacts>

    Scenes without brands are left out.

    Request throttling:
    Applies the rate limit set in the config (section `video_db`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video_id: The VideoDB video ID, as returned by `upload()`.
        collection_id: The VideoDB collection that contains the video.
        model: The VideoDB model tier, such as `mini`, `basic`, `pro` or `ultra`. If not specified, VideoDB uses
            its default.
        sampling: Which frames the model sees in each scene, such as `{'strategy': 'uniform', 'frame_count': 4}`.
        segmentation: How to split the video into scenes, such as `{'type': 'time', 'seconds': 10}` or
            `{'type': 'shot', 'threshold': 30}`. If not specified, VideoDB uses its default.
        transform: Preprocessing before analysis, such as `{'resolution': '480p'}` to reduce cost.

    Returns:
        A list of scenes with the brands found in each, and why each brand was found.

    Examples:
        >>> tbl.add_computed_column(brands=detect_brands(tbl.videodb_id))
    """
    rows = await _run_visual('brand_detection', video_id, collection_id, model, sampling, segmentation, transform)
    return [
        SceneBrands(
            start=start,
            end=end,
            brand_names=data.get('brand_names') or [],
            mentions=[
                BrandMention(brand_name=m.get('brand_name') or '', reason=m.get('reason') or '')
                for m in data.get('mentions') or []
            ],
        )
        for start, end, data in rows
    ]


@pxt.udf(is_deterministic=False, resource_pool='request-rate:video_db')
async def recognize_activity(
    video_id: str,
    *,
    collection_id: str = 'default',
    model: str | None = None,
    sampling: dict[str, Any] | None = None,
    segmentation: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
) -> list[SceneActivity]:
    """
    Names the main action or activity in each scene of a VideoDB video, such as `walking` or `cooking`.

    Runs the VideoDB `activity_recognition` analyzer through `Video.understand()`:
    <https://docs.videodb.io/pages/understand/indexing-pipelines/understanding-artifacts>

    Request throttling:
    Applies the rate limit set in the config (section `video_db`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video_id: The VideoDB video ID, as returned by `upload()`.
        collection_id: The VideoDB collection that contains the video.
        model: The VideoDB model tier, such as `mini`, `basic`, `pro` or `ultra`. If not specified, VideoDB uses
            its default.
        sampling: Which frames the model sees in each scene, such as `{'strategy': 'uniform', 'frame_count': 4}`.
        segmentation: How to split the video into scenes, such as `{'type': 'time', 'seconds': 10}` or
            `{'type': 'shot', 'threshold': 30}`. If not specified, VideoDB uses its default.
        transform: Preprocessing before analysis, such as `{'resolution': '480p'}` to reduce cost.

    Returns:
        A list of scenes with their activity.

    Examples:
        >>> tbl.add_computed_column(activity=recognize_activity(tbl.videodb_id))
    """
    rows = await _run_visual('activity_recognition', video_id, collection_id, model, sampling, segmentation, transform)
    return [SceneActivity(start=start, end=end, activity=data.get('activity') or '') for start, end, data in rows]


@pxt.udf(is_deterministic=False, resource_pool='request-rate:video_db')
async def detect_location(
    video_id: str,
    *,
    collection_id: str = 'default',
    model: str | None = None,
    sampling: dict[str, Any] | None = None,
    segmentation: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
) -> list[SceneLocation]:
    """
    Describes the setting of each scene of a VideoDB video, such as an indoor office or a city street.

    Runs the VideoDB `location_detection` analyzer through `Video.understand()`:
    <https://docs.videodb.io/pages/understand/indexing-pipelines/understanding-artifacts>

    Request throttling:
    Applies the rate limit set in the config (section `video_db`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video_id: The VideoDB video ID, as returned by `upload()`.
        collection_id: The VideoDB collection that contains the video.
        model: The VideoDB model tier, such as `mini`, `basic`, `pro` or `ultra`. If not specified, VideoDB uses
            its default.
        sampling: Which frames the model sees in each scene, such as `{'strategy': 'uniform', 'frame_count': 4}`.
        segmentation: How to split the video into scenes, such as `{'type': 'time', 'seconds': 10}` or
            `{'type': 'shot', 'threshold': 30}`. If not specified, VideoDB uses its default.
        transform: Preprocessing before analysis, such as `{'resolution': '480p'}` to reduce cost.

    Returns:
        A list of scenes with their location.

    Examples:
        >>> tbl.add_computed_column(location=detect_location(tbl.videodb_id))
    """
    rows = await _run_visual('location_detection', video_id, collection_id, model, sampling, segmentation, transform)
    return [SceneLocation(start=start, end=end, location=data.get('location') or '') for start, end, data in rows]


@pxt.udf(is_deterministic=False, resource_pool='request-rate:video_db')
async def detect_objects(
    video_id: str,
    *,
    collection_id: str = 'default',
    model: str | None = None,
    sandbox_id: str | None = None,
    sampling: dict[str, Any] | None = None,
    segmentation: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
) -> list[FrameObjects]:
    """
    Detects objects with bounding boxes in sampled frames of a VideoDB video, using the COCO label vocabulary.

    Runs the VideoDB `object_detection` analyzer through `Video.understand()`:
    <https://docs.videodb.io/pages/understand/indexing-pipelines/understanding-artifacts>

    The default object detection model runs on a VideoDB sandbox
    (<https://docs.videodb.io/pages/core-concepts/sandbox-compute>). Start one with
    `model_categories=['object_detection']` and pass its ID as `sandbox_id`.

    Frames without objects are left out.

    Request throttling:
    Applies the rate limit set in the config (section `video_db`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video_id: The VideoDB video ID, as returned by `upload()`.
        collection_id: The VideoDB collection that contains the video.
        model: The object detection model, such as `rtdetr-v2-r50vd`. If not specified, VideoDB uses its default.
        sandbox_id: The ID of the active VideoDB sandbox to run the model on.
        sampling: Which frames to analyze, such as `{'strategy': 'interval', 'every': 1}` for one frame per second.
        segmentation: How to split the video into scenes, such as `{'type': 'time', 'seconds': 10}`.
        transform: Preprocessing before analysis, such as `{'resolution': '480p'}` to reduce cost.

    Returns:
        A list of frames with the objects found in each.

    Examples:
        >>> tbl.add_computed_column(
        ...     objects=detect_objects(
        ...         tbl.videodb_id,
        ...         sandbox_id='...',
        ...         sampling={'strategy': 'interval', 'every': 1},
        ...     )
        ... )
    """
    Env.get().require_package('videodb')
    analyzer = _analyzer('object_detection', config={'model': model, 'sandbox_id': sandbox_id}, sampling=sampling)
    scenes = await asyncio.to_thread(_understand, video_id, collection_id, [analyzer], segmentation, transform)
    frames = []
    for scene in scenes:
        for frame in (scene.get('data') or {}).get('frames') or []:
            detections = [
                ObjectDetection(
                    label=d['label'],
                    score=float(d.get('score') or 0.0),
                    box=[float(v) for v in d['box']['box']],
                    unit=d['box'].get('unit') or 'normalized',
                )
                for d in frame.get('detections') or []
            ]
            if detections:
                frames.append(FrameObjects(timestamp=float(frame['timestamp']), detections=detections))
    return frames


__all__ = [
    'BrandMention',
    'FrameObjects',
    'ObjectDetection',
    'OnScreenText',
    'SceneActivity',
    'SceneBrands',
    'SceneDescription',
    'SceneLocation',
    'TranscriptSegment',
    'describe_scenes',
    'detect_brands',
    'detect_location',
    'detect_objects',
    'read_text',
    'recognize_activity',
    'transcribe',
    'upload',
]


def __dir__() -> list[str]:
    return __all__
