using System;
using System.Collections;
using System.Globalization;
using System.Text;
using Meta.XR;
using UnityEngine;
using UnityEngine.Networking;

namespace EyeTracking.Recording
{
    [DisallowMultipleComponent]
    public sealed class QuestPcCalibrationRecorder : MonoBehaviour
    {
        public const string ProtocolName = "quest_pc_calibration_recording_v1";

        [Header("Camera Access")]
        [SerializeField] private QuestCameraRecorder recorder;
        [SerializeField] private PassthroughCameraAccess leftCameraAccess;
        [SerializeField] private PassthroughCameraAccess rightCameraAccess;
        [SerializeField] private bool recordRightCamera = true;

        [Header("PC Receiver")]
        [SerializeField] private string serverUrl = "http://10.128.1.15:9101";
        [SerializeField, Min(0.05f)] private float frameIntervalSeconds = 1f / 15f;
        [SerializeField, Range(10, 100)] private int jpegQuality = 75;
        [SerializeField] private bool flipVertical = true;
        [SerializeField] private bool logStatus = true;

        private Texture2D readbackTexture;
        private RenderTexture scratchRenderTexture;
        private bool isRecording;
        private bool isStarting;
        private bool stopRequestedWhileStarting;
        private bool requestInFlight;
        private float nextFrameTime;
        private string currentRecordId;
        private DateTime startTimeUtc;
        private double recordingStartRealtimeSeconds;
        private int leftFrameIndex;
        private int rightFrameIndex;
        private int sampleIndex;

        public bool IsRecording => isRecording || isStarting;
        public string CurrentRecordId => currentRecordId;

        private void Awake()
        {
            ResolveReferences();
        }

        private void Update()
        {
            if (!isRecording || requestInFlight || Time.unscaledTime < nextFrameTime)
            {
                return;
            }

            nextFrameTime = Time.unscaledTime + frameIntervalSeconds;
            StartCoroutine(SendFrameSet());
        }

        public void ToggleCalibrationRecording()
        {
            if (IsRecording)
            {
                StopCalibrationRecording();
            }
            else
            {
                StartCalibrationRecording();
            }
        }

        public void StartCalibrationRecording()
        {
            if (isRecording || isStarting)
            {
                return;
            }

            StartCoroutine(StartCalibrationRecordingRoutine());
        }

        public void StopCalibrationRecording()
        {
            if (isStarting)
            {
                stopRequestedWhileStarting = true;
                return;
            }

            if (!isRecording)
            {
                return;
            }

            StartCoroutine(StopCalibrationRecordingRoutine());
        }

        private IEnumerator StartCalibrationRecordingRoutine()
        {
            isStarting = true;
            stopRequestedWhileStarting = false;
            ResolveReferences();

            if (!IsCameraPlayable(leftCameraAccess))
            {
                Debug.LogWarning("[QuestPcCalibrationRecorder] Cannot start PC calibration: left camera is not ready.", this);
                isStarting = false;
                yield break;
            }

            bool hasRight = recordRightCamera && IsCameraPlayable(rightCameraAccess);
            startTimeUtc = DateTime.UtcNow;
            recordingStartRealtimeSeconds = Time.realtimeSinceStartupAsDouble;
            currentRecordId = $"record_pc_calib_{DateTime.Now:yyyyMMdd_HHmmss}";
            leftFrameIndex = 0;
            rightFrameIndex = 0;
            sampleIndex = 0;
            nextFrameTime = 0f;

            CalibrationStartMessage start = new CalibrationStartMessage
            {
                protocol = ProtocolName,
                type = "calibration_start",
                recordId = currentRecordId,
                startTimeUtc = startTimeUtc.ToString("o", CultureInfo.InvariantCulture),
                metadata = BuildMetadata(hasRight)
            };

            yield return PostJson("calibration/start", JsonUtility.ToJson(start, false), ok =>
            {
                if (!ok)
                {
                    currentRecordId = null;
                    return;
                }

                isRecording = true;
                nextFrameTime = 0f;
                if (logStatus)
                {
                    Debug.Log("[QuestPcCalibrationRecorder] PC calibration recording started: " + currentRecordId, this);
                }
            });

            isStarting = false;

            if (stopRequestedWhileStarting)
            {
                StopCalibrationRecording();
            }
        }

        private IEnumerator StopCalibrationRecordingRoutine()
        {
            string recordId = currentRecordId;
            isRecording = false;

            while (requestInFlight)
            {
                yield return null;
            }

            CalibrationStopMessage stop = new CalibrationStopMessage
            {
                protocol = ProtocolName,
                type = "calibration_stop",
                recordId = recordId,
                unityTimestampSeconds = Time.realtimeSinceStartupAsDouble,
                durationSeconds = Time.realtimeSinceStartupAsDouble - recordingStartRealtimeSeconds
            };

            yield return PostJson("calibration/stop", JsonUtility.ToJson(stop, false), _ => { });

            if (logStatus)
            {
                Debug.Log("[QuestPcCalibrationRecorder] PC calibration recording stopped: " + recordId, this);
            }

            currentRecordId = null;
        }

        private IEnumerator SendFrameSet()
        {
            requestInFlight = true;
            int leftIndex = -1;
            int rightIndex = -1;

            ResolveReferences();
            if (IsCameraPlayable(leftCameraAccess))
            {
                leftIndex = leftFrameIndex++;
                yield return SendCameraFrame(leftCameraAccess, "left", leftIndex);
            }

            if (recordRightCamera && IsCameraPlayable(rightCameraAccess))
            {
                rightIndex = rightFrameIndex++;
                yield return SendCameraFrame(rightCameraAccess, "right", rightIndex);
            }

            if (recorder != null)
            {
                QuestCameraRecorder.TrajectorySample sample = recorder.CaptureExternalTrajectorySample(
                    sampleIndex++,
                    leftIndex,
                    rightIndex,
                    Time.realtimeSinceStartupAsDouble - recordingStartRealtimeSeconds,
                    "pc_calibration_recording");

                CalibrationSampleMessage sampleMessage = new CalibrationSampleMessage
                {
                    protocol = ProtocolName,
                    type = "calibration_sample",
                    recordId = currentRecordId,
                    sample = sample
                };

                yield return PostJson("calibration/sample", JsonUtility.ToJson(sampleMessage, false), _ => { });
            }

            requestInFlight = false;
        }

        private IEnumerator SendCameraFrame(PassthroughCameraAccess cameraAccess, string side, int frameIndex)
        {
            Pose pose = cameraAccess.GetCameraPose();
            double unityTimestampSeconds = Time.realtimeSinceStartupAsDouble;
            double cameraTimestampSeconds = new DateTimeOffset(cameraAccess.Timestamp).ToUnixTimeMilliseconds() / 1000.0;
            Texture texture = cameraAccess.GetTexture();
            Texture2D readable = TextureToReadableTexture2D(texture, cameraAccess.CurrentResolution);
            if (readable == null)
            {
                yield break;
            }

            byte[] jpg = readable.EncodeToJPG(jpegQuality);
            string url = BuildFrameUrl(cameraAccess, side, frameIndex, unityTimestampSeconds, cameraTimestampSeconds, pose);
            using UnityWebRequest request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST);
            request.uploadHandler = new UploadHandlerRaw(jpg);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "image/jpeg");
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning("[QuestPcCalibrationRecorder] Frame post failed: " + request.error, this);
            }
        }

        private string BuildFrameUrl(
            PassthroughCameraAccess cameraAccess,
            string side,
            int frameIndex,
            double unityTimestampSeconds,
            double cameraTimestampSeconds,
            Pose pose)
        {
            Vector2Int resolution = cameraAccess.CurrentResolution;
            return serverUrl.TrimEnd('/') +
                "/calibration/frame?recordId=" + Escape(currentRecordId) +
                "&side=" + Escape(side) +
                "&frameIndex=" + frameIndex.ToString(CultureInfo.InvariantCulture) +
                "&unityTimestampSeconds=" + unityTimestampSeconds.ToString("R", CultureInfo.InvariantCulture) +
                "&cameraTimestampSeconds=" + cameraTimestampSeconds.ToString("R", CultureInfo.InvariantCulture) +
                "&pose=" + Escape(Join(PoseToArray(pose))) +
                "&width=" + resolution.x.ToString(CultureInfo.InvariantCulture) +
                "&height=" + resolution.y.ToString(CultureInfo.InvariantCulture) +
                "&flipVertical=" + (flipVertical ? "1" : "0");
        }

        private IEnumerator PostJson(string relativePath, string json, Action<bool> onComplete)
        {
            string url = serverUrl.TrimEnd('/') + "/" + relativePath.TrimStart('/');
            byte[] body = Encoding.UTF8.GetBytes(json);
            using UnityWebRequest request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST);
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            yield return request.SendWebRequest();

            bool ok = request.result == UnityWebRequest.Result.Success;
            if (!ok)
            {
                Debug.LogWarning("[QuestPcCalibrationRecorder] JSON post failed: " + request.error + " url=" + url, this);
            }

            onComplete?.Invoke(ok);
        }

        private CalibrationMetadata BuildMetadata(bool hasRight)
        {
            return new CalibrationMetadata
            {
                schemaVersion = "quest_pc_calibration_recorder_metadata_v2",
                startTimeUtc = startTimeUtc.ToString("o", CultureInfo.InvariantCulture),
                outputDirectory = "pc_receiver_raw",
                durationSeconds = 0.0,
                fps = Mathf.Max(1, Mathf.RoundToInt(1f / Mathf.Max(0.001f, frameIntervalSeconds))),
                poseConvention = "xyzqwqxqyqz",
                intrinsicsConvention = "fxfycxcy",
                trajectoryFrame = "unity_world",
                trajectorySampleSource = "Quest sends passthrough JPEG frames and trajectory samples to the PC receiver; PC writes the raw calibration record.",
                trajectoryContents = "left/right passthrough camera pose, HMD eye poses, gaze ray, EyeGazePose visual-object gaze point, optional EnvironmentMapper hit point, and controller poses in Unity world frame when available",
                cameraPoseConvention = "PassthroughCameraAccess.GetCameraPose in Unity world frame",
                gazePointConvention = "gazePoint3DWorld is EyeGazePose.vizObj Transform.position in Unity world frame; gazePointWorld is the EnvironmentMapper depth hit when available.",
                gazeProjectionConvention = "left/rightGazePointViewport are normalized camera viewport coordinates with origin bottom-left. left/rightGazePointPixel are recorded-video pixel coordinates with origin top-left; recorder flipVertical is already applied.",
                cameraDistortionModel = "pinhole_no_distortion",
                realtimeTelemetryProtocol = ProtocolName,
                realtimeTelemetryContents = "HTTP JSON/JPEG: calibration_start, calibration/frame, calibration_sample, calibration_stop.",
                recordSynchronizedTrajectory = true,
                recordHmdEyePoses = true,
                hasLeftCameraAccess = leftCameraAccess != null,
                hasRightCameraAccess = rightCameraAccess != null,
                recordRightCamera = hasRight,
                leftVideoFileName = "left_recording.mp4",
                rightVideoFileName = hasRight ? "right_recording.mp4" : null,
                leftFrameMetadataFileName = "left_frames.jsonl",
                rightFrameMetadataFileName = hasRight ? "right_frames.jsonl" : null,
                trajectoryFileName = "trajectory.jsonl",
                leftIntrinsics = IntrinsicsToArray(leftCameraAccess),
                rightIntrinsics = hasRight ? IntrinsicsToArray(rightCameraAccess) : null,
                leftResolution = ResolutionToArray(leftCameraAccess),
                rightResolution = hasRight ? ResolutionToArray(rightCameraAccess) : null
            };
        }

        private void ResolveReferences()
        {
            if (recorder == null)
            {
                recorder = GetComponent<QuestCameraRecorder>() ?? FindFirstObjectByType<QuestCameraRecorder>();
            }

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

        private static bool IsCameraPlayable(PassthroughCameraAccess cameraAccess)
        {
            return cameraAccess != null && cameraAccess.IsPlaying;
        }

        private static double[] IntrinsicsToArray(PassthroughCameraAccess cameraAccess)
        {
            if (cameraAccess == null)
            {
                return null;
            }

            PassthroughCameraAccess.CameraIntrinsics intrinsics = cameraAccess.Intrinsics;
            Vector2Int resolution = cameraAccess.CurrentResolution;
            Vector2Int sensorResolution = intrinsics.SensorResolution;
            return new[]
            {
                (double)intrinsics.FocalLength.x * resolution.x / sensorResolution.x,
                (double)intrinsics.FocalLength.y * resolution.y / sensorResolution.y,
                (double)intrinsics.PrincipalPoint.x * resolution.x / sensorResolution.x,
                (double)intrinsics.PrincipalPoint.y * resolution.y / sensorResolution.y
            };
        }

        private static int[] ResolutionToArray(PassthroughCameraAccess cameraAccess)
        {
            if (cameraAccess == null)
            {
                return null;
            }

            Vector2Int resolution = cameraAccess.CurrentResolution;
            return new[] { resolution.x, resolution.y };
        }

        private static double[] PoseToArray(Pose pose)
        {
            return new[]
            {
                (double)pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.rotation.w,
                pose.rotation.x,
                pose.rotation.y,
                pose.rotation.z
            };
        }

        private static string Join(double[] values)
        {
            string[] parts = new string[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                parts[i] = values[i].ToString("R", CultureInfo.InvariantCulture);
            }

            return string.Join(",", parts);
        }

        private static string Escape(string value)
        {
            return UnityWebRequest.EscapeURL(value ?? string.Empty);
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

        [Serializable]
        private sealed class CalibrationStartMessage
        {
            public string protocol;
            public string type;
            public string recordId;
            public string startTimeUtc;
            public CalibrationMetadata metadata;
        }

        [Serializable]
        private sealed class CalibrationStopMessage
        {
            public string protocol;
            public string type;
            public string recordId;
            public double unityTimestampSeconds;
            public double durationSeconds;
        }

        [Serializable]
        private sealed class CalibrationSampleMessage
        {
            public string protocol;
            public string type;
            public string recordId;
            public QuestCameraRecorder.TrajectorySample sample;
        }

        [Serializable]
        private sealed class CalibrationMetadata
        {
            public string schemaVersion;
            public string startTimeUtc;
            public string outputDirectory;
            public double durationSeconds;
            public int fps;
            public string poseConvention;
            public string intrinsicsConvention;
            public string trajectoryFrame;
            public string trajectorySampleSource;
            public string trajectoryContents;
            public string cameraPoseConvention;
            public string gazePointConvention;
            public string gazeProjectionConvention;
            public string cameraDistortionModel;
            public string realtimeTelemetryProtocol;
            public string realtimeTelemetryContents;
            public bool recordSynchronizedTrajectory;
            public bool recordHmdEyePoses;
            public bool hasLeftCameraAccess;
            public bool hasRightCameraAccess;
            public bool recordRightCamera;
            public string leftVideoFileName;
            public string rightVideoFileName;
            public string leftFrameMetadataFileName;
            public string rightFrameMetadataFileName;
            public string trajectoryFileName;
            public double[] leftIntrinsics;
            public double[] rightIntrinsics;
            public int[] leftResolution;
            public int[] rightResolution;
        }
    }
}
