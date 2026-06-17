using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Text;
using Meta.XR;
using UnityEngine;
using UnityEngine.Networking;

namespace EyeTracking.Recording
{
    [DisallowMultipleComponent]
    public sealed class QuestPcaCheckerboardRelay : MonoBehaviour
    {
        private const string ProtocolName = "quest_pca_checkerboard_relay_v1";

        [Header("Camera Access")]
        [SerializeField] private PassthroughCameraAccess leftCameraAccess;
        [SerializeField] private PassthroughCameraAccess rightCameraAccess;
        [SerializeField] private bool sendLeft = true;
        [SerializeField] private bool sendRight = true;

        [Header("PC Relay")]
        [SerializeField] private bool relayEnabled;
        [SerializeField] private string serverUrl = "http://10.128.0.227:9101";
        [SerializeField, Min(0.05f)] private float intervalSeconds = 0.5f;
        [SerializeField, Range(10, 100)] private int jpegQuality = 70;
        [SerializeField] private bool viewportFlipY = true;
        [SerializeField, Min(0.1f)] private float roundTripDistanceMeters = 1.0f;
        [SerializeField] private bool logStatus = true;

        [Header("File Commands")]
        [SerializeField] private bool enableFileCommands = true;
        [SerializeField] private string commandFileName = "pca_relay_command.txt";
        [SerializeField, Min(0.1f)] private float commandPollIntervalSeconds = 0.25f;
        [SerializeField] private bool deleteCommandAfterRead = true;

        private Texture2D readbackTexture;
        private RenderTexture scratchRenderTexture;
        private bool requestInFlight;
        private float nextSendTime;
        private float nextCommandPollTime;
        private int sequence;
        private bool nextAutoSendLeft = true;
        private string commandPath;

        private void Awake()
        {
            commandPath = Path.Combine(Application.persistentDataPath, commandFileName);
            ResolveCameraAccesses();
            if (logStatus)
            {
                Debug.Log("[QuestPcaCheckerboardRelay] Command path: " + commandPath, this);
            }
        }

        private void Update()
        {
            PollCommandFile();

            if (!relayEnabled || requestInFlight || Time.unscaledTime < nextSendTime)
            {
                return;
            }

            nextSendTime = Time.unscaledTime + intervalSeconds;
            ResolveCameraAccesses();

            PassthroughCameraAccess cameraAccess;
            string side;
            if (!TryGetNextCameraForAutoSend(out cameraAccess, out side))
            {
                return;
            }

            StartCoroutine(SendCameraFrame(cameraAccess, side, sequence++));
        }

        public void StartRelay(string newServerUrl = null)
        {
            if (!string.IsNullOrWhiteSpace(newServerUrl))
            {
                serverUrl = newServerUrl.Trim();
            }

            relayEnabled = true;
            nextSendTime = 0f;
            if (logStatus)
            {
                Debug.Log("[QuestPcaCheckerboardRelay] Relay started: " + serverUrl, this);
            }
        }

        public void StopRelay()
        {
            relayEnabled = false;
            if (logStatus)
            {
                Debug.Log("[QuestPcaCheckerboardRelay] Relay stopped.", this);
            }
        }

        private IEnumerator SendCameraFrame(PassthroughCameraAccess cameraAccess, string side, int requestSequence)
        {
            requestInFlight = true;
            Pose cachedPose = cameraAccess.GetCameraPose();
            Texture texture = cameraAccess.GetTexture();
            Texture2D readable = TextureToReadableTexture2D(texture, cameraAccess.CurrentResolution);
            if (readable == null)
            {
                requestInFlight = false;
                yield break;
            }

            byte[] jpg = readable.EncodeToJPG(jpegQuality);
            string detectUrl = BuildDetectUrl(cameraAccess, side, requestSequence);
            CornerDetectionResponse response = null;
            using (UnityWebRequest request = new UnityWebRequest(detectUrl, UnityWebRequest.kHttpVerbPOST))
            {
                request.uploadHandler = new UploadHandlerRaw(jpg);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "image/jpeg");
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning("[QuestPcaCheckerboardRelay] Detect request failed: " + request.error, this);
                    requestInFlight = false;
                    yield break;
                }

                try
                {
                    response = JsonUtility.FromJson<CornerDetectionResponse>(request.downloadHandler.text);
                }
                catch (Exception exception)
                {
                    Debug.LogWarning("[QuestPcaCheckerboardRelay] Could not parse detect response: " + exception.Message, this);
                    requestInFlight = false;
                    yield break;
                }
            }

            OfficialPcaResult result = BuildOfficialResult(cameraAccess, side, requestSequence, cachedPose, response);
            yield return PostOfficialResult(result);
            requestInFlight = false;
        }

        private string BuildDetectUrl(PassthroughCameraAccess cameraAccess, string side, int requestSequence)
        {
            string baseUrl = serverUrl.TrimEnd('/');
            double cameraTimestampSeconds = cameraAccess.Timestamp.ToUniversalTime().Subtract(DateTime.UnixEpoch).TotalSeconds;
            return baseUrl +
                "/detect?side=" + UnityWebRequest.EscapeURL(side) +
                "&sequence=" + requestSequence.ToString(CultureInfo.InvariantCulture) +
                "&unityTimestampSeconds=" + Time.realtimeSinceStartupAsDouble.ToString("R", CultureInfo.InvariantCulture) +
                "&cameraTimestampSeconds=" + cameraTimestampSeconds.ToString("R", CultureInfo.InvariantCulture) +
                "&viewportFlipY=" + (viewportFlipY ? "1" : "0");
        }

        private OfficialPcaResult BuildOfficialResult(
            PassthroughCameraAccess cameraAccess,
            string side,
            int requestSequence,
            Pose cachedPose,
            CornerDetectionResponse response)
        {
            CameraApiSnapshot camera = BuildCameraSnapshot(cameraAccess, cachedPose);
            OfficialPcaResult result = new OfficialPcaResult
            {
                protocol = ProtocolName,
                type = "official_pca_corner_rays",
                ok = response != null && response.ok,
                error = response != null ? response.error : "missing_response",
                sequence = requestSequence,
                side = side,
                unityTimestampSeconds = Time.realtimeSinceStartupAsDouble,
                camera = camera,
                detection = response,
                corners = Array.Empty<OfficialCornerResult>()
            };

            if (response == null || response.corners == null || response.corners.Length == 0)
            {
                return result;
            }

            OfficialCornerResult[] corners = new OfficialCornerResult[response.corners.Length];
            for (int i = 0; i < response.corners.Length; i++)
            {
                DetectedCorner corner = response.corners[i];
                Vector2 viewport = ArrayToVector2(corner.viewport);
                Ray ray = cameraAccess.ViewportPointToRay(viewport, cachedPose);
                Vector3 testPoint = ray.origin + ray.direction.normalized * roundTripDistanceMeters;
                Vector2 roundTrip = cameraAccess.WorldToViewportPoint(testPoint, cachedPose);
                corners[i] = new OfficialCornerResult
                {
                    index = corner.index,
                    pixel = corner.pixel,
                    viewport = Vector2ToArray(viewport),
                    rayOrigin = Vector3ToArray(ray.origin),
                    rayDirection = Vector3ToArray(ray.direction.normalized),
                    roundTripWorldPoint = Vector3ToArray(testPoint),
                    roundTripViewport = Vector2ToArray(roundTrip),
                    roundTripViewportError = Vector2ToArray(roundTrip - viewport)
                };
            }

            result.corners = corners;
            return result;
        }

        private IEnumerator PostOfficialResult(OfficialPcaResult result)
        {
            string json = JsonUtility.ToJson(result, false);
            byte[] body = Encoding.UTF8.GetBytes(json);
            string url = serverUrl.TrimEnd('/') + "/official_result";
            using (UnityWebRequest request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST))
            {
                request.uploadHandler = new UploadHandlerRaw(body);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning("[QuestPcaCheckerboardRelay] Result post failed: " + request.error, this);
                }
                else if (logStatus)
                {
                    int count = result.corners != null ? result.corners.Length : 0;
                    Debug.Log(
                        "[QuestPcaCheckerboardRelay] Posted official PCA result " +
                        $"seq={result.sequence} side={result.side} ok={result.ok} corners={count}",
                        this);
                }
            }
        }

        private CameraApiSnapshot BuildCameraSnapshot(PassthroughCameraAccess cameraAccess, Pose cachedPose)
        {
            PassthroughCameraAccess.CameraIntrinsics intrinsics = cameraAccess.Intrinsics;
            return new CameraApiSnapshot
            {
                currentResolution = Vector2IntToArray(cameraAccess.CurrentResolution),
                focalLength = Vector2ToArray(intrinsics.FocalLength),
                principalPoint = Vector2ToArray(intrinsics.PrincipalPoint),
                sensorResolution = Vector2IntToArray(intrinsics.SensorResolution),
                lensOffset = PoseToArray(intrinsics.LensOffset),
                cameraPose = PoseToArray(cachedPose)
            };
        }

        private Texture2D TextureToReadableTexture2D(Texture texture, Vector2Int resolution)
        {
            if (texture == null || resolution.x <= 0 || resolution.y <= 0)
            {
                return null;
            }

            if (readbackTexture == null ||
                readbackTexture.width != resolution.x ||
                readbackTexture.height != resolution.y)
            {
                if (readbackTexture != null)
                {
                    Destroy(readbackTexture);
                }

                readbackTexture = new Texture2D(resolution.x, resolution.y, TextureFormat.RGBA32, false);
            }

            RenderTexture previous = RenderTexture.active;
            try
            {
                RenderTexture sourceRenderTexture = texture as RenderTexture;
                if (sourceRenderTexture == null)
                {
                    EnsureScratchRenderTexture(resolution);
                    Graphics.Blit(texture, scratchRenderTexture);
                    sourceRenderTexture = scratchRenderTexture;
                }

                RenderTexture.active = sourceRenderTexture;
                readbackTexture.ReadPixels(new Rect(0, 0, resolution.x, resolution.y), 0, 0, false);
                readbackTexture.Apply(false, false);
                return readbackTexture;
            }
            finally
            {
                RenderTexture.active = previous;
            }
        }

        private void EnsureScratchRenderTexture(Vector2Int resolution)
        {
            if (scratchRenderTexture != null &&
                scratchRenderTexture.width == resolution.x &&
                scratchRenderTexture.height == resolution.y)
            {
                return;
            }

            if (scratchRenderTexture != null)
            {
                scratchRenderTexture.Release();
                Destroy(scratchRenderTexture);
            }

            scratchRenderTexture = new RenderTexture(resolution.x, resolution.y, 0, RenderTextureFormat.ARGB32);
            scratchRenderTexture.Create();
        }

        private void PollCommandFile()
        {
            if (!enableFileCommands || Time.unscaledTime < nextCommandPollTime)
            {
                return;
            }

            nextCommandPollTime = Time.unscaledTime + commandPollIntervalSeconds;
            if (string.IsNullOrEmpty(commandPath) || !File.Exists(commandPath))
            {
                return;
            }

            string commandText;
            try
            {
                commandText = File.ReadAllText(commandPath, Encoding.UTF8).Trim();
                if (deleteCommandAfterRead)
                {
                    File.Delete(commandPath);
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning("[QuestPcaCheckerboardRelay] Could not read command file: " + exception.Message, this);
                return;
            }

            ExecuteCommand(commandText);
        }

        private void ExecuteCommand(string commandText)
        {
            if (string.IsNullOrWhiteSpace(commandText))
            {
                return;
            }

            string[] parts = commandText.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            string command = parts.Length > 0 ? parts[0].ToLowerInvariant() : string.Empty;
            if (command == "start")
            {
                StartRelay(parts.Length > 1 ? parts[1] : null);
                return;
            }

            if (command == "stop")
            {
                StopRelay();
                return;
            }

            if (command == "once")
            {
                if (requestInFlight)
                {
                    if (logStatus)
                    {
                        Debug.Log("[QuestPcaCheckerboardRelay] One-shot ignored; request already in flight.", this);
                    }
                    return;
                }

                StartCoroutine(SendOneShot());
                return;
            }

            Debug.LogWarning("[QuestPcaCheckerboardRelay] Unknown command: " + commandText, this);
        }

        private IEnumerator SendOneShot()
        {
            if (requestInFlight)
            {
                yield break;
            }

            ResolveCameraAccesses();
            if (IsCameraPlayable(leftCameraAccess))
            {
                yield return SendCameraFrame(leftCameraAccess, "left", sequence++);
            }
            if (IsCameraPlayable(rightCameraAccess))
            {
                yield return SendCameraFrame(rightCameraAccess, "right", sequence++);
            }
        }

        private bool TryGetNextCameraForAutoSend(out PassthroughCameraAccess cameraAccess, out string side)
        {
            bool canSendLeft = sendLeft && IsCameraPlayable(leftCameraAccess);
            bool canSendRight = sendRight && IsCameraPlayable(rightCameraAccess);

            if (canSendLeft && canSendRight)
            {
                bool chooseLeft = nextAutoSendLeft;
                nextAutoSendLeft = !nextAutoSendLeft;
                cameraAccess = chooseLeft ? leftCameraAccess : rightCameraAccess;
                side = chooseLeft ? "left" : "right";
                return true;
            }

            if (canSendLeft)
            {
                nextAutoSendLeft = false;
                cameraAccess = leftCameraAccess;
                side = "left";
                return true;
            }

            if (canSendRight)
            {
                nextAutoSendLeft = true;
                cameraAccess = rightCameraAccess;
                side = "right";
                return true;
            }

            cameraAccess = null;
            side = null;
            return false;
        }

        private static bool IsCameraPlayable(PassthroughCameraAccess cameraAccess)
        {
            return cameraAccess != null && cameraAccess.IsPlaying;
        }

        private void ResolveCameraAccesses()
        {
            if (leftCameraAccess != null && rightCameraAccess != null)
            {
                return;
            }

            PassthroughCameraAccess[] accesses = FindObjectsByType<PassthroughCameraAccess>(FindObjectsSortMode.None);
            foreach (PassthroughCameraAccess access in accesses)
            {
                if (access == null)
                {
                    continue;
                }

                if (access.CameraPosition == PassthroughCameraAccess.CameraPositionType.Left && leftCameraAccess == null)
                {
                    leftCameraAccess = access;
                }
                else if (access.CameraPosition == PassthroughCameraAccess.CameraPositionType.Right && rightCameraAccess == null)
                {
                    rightCameraAccess = access;
                }
            }
        }

        private void OnDestroy()
        {
            if (readbackTexture != null)
            {
                Destroy(readbackTexture);
            }

            if (scratchRenderTexture != null)
            {
                scratchRenderTexture.Release();
                Destroy(scratchRenderTexture);
            }
        }

        private static Vector2 ArrayToVector2(float[] value)
        {
            return value != null && value.Length >= 2
                ? new Vector2(value[0], value[1])
                : Vector2.zero;
        }

        private static float[] Vector2ToArray(Vector2 value)
        {
            return new[] { value.x, value.y };
        }

        private static float[] Vector3ToArray(Vector3 value)
        {
            return new[] { value.x, value.y, value.z };
        }

        private static int[] Vector2IntToArray(Vector2Int value)
        {
            return new[] { value.x, value.y };
        }

        private static float[] PoseToArray(Pose pose)
        {
            return new[]
            {
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.rotation.w,
                pose.rotation.x,
                pose.rotation.y,
                pose.rotation.z
            };
        }

        [Serializable]
        private sealed class CornerDetectionResponse
        {
            public string protocol;
            public bool ok;
            public string error;
            public int sequence;
            public string side;
            public double unityTimestampSeconds;
            public double cameraTimestampSeconds;
            public int imageWidth;
            public int imageHeight;
            public int[] pattern;
            public int cornerCount;
            public bool viewportFlipY;
            public string pixelOrigin;
            public string viewportOrigin;
            public DetectedCorner[] corners;
        }

        [Serializable]
        private sealed class DetectedCorner
        {
            public int index;
            public float[] pixel;
            public float[] viewport;
        }

        [Serializable]
        private sealed class OfficialPcaResult
        {
            public string protocol;
            public string type;
            public bool ok;
            public string error;
            public int sequence;
            public string side;
            public double unityTimestampSeconds;
            public CameraApiSnapshot camera;
            public CornerDetectionResponse detection;
            public OfficialCornerResult[] corners;
        }

        [Serializable]
        private sealed class CameraApiSnapshot
        {
            public int[] currentResolution;
            public float[] focalLength;
            public float[] principalPoint;
            public int[] sensorResolution;
            public float[] lensOffset;
            public float[] cameraPose;
        }

        [Serializable]
        private sealed class OfficialCornerResult
        {
            public int index;
            public float[] pixel;
            public float[] viewport;
            public float[] rayOrigin;
            public float[] rayDirection;
            public float[] roundTripWorldPoint;
            public float[] roundTripViewport;
            public float[] roundTripViewportError;
        }
    }
}
