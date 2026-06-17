using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Anaglyph.XRTemplate;
using Meta.XR;
using UnityEngine;

namespace EyeTracking.Recording
{
    public sealed class QuestCameraRecorder : MonoBehaviour
    {
        [Header("Camera Access")]
        [SerializeField] private PassthroughCameraAccess leftCameraAccess;
        [SerializeField] private PassthroughCameraAccess rightCameraAccess;
        [SerializeField] private bool recordRightCamera;

        [Header("Recording")]
        [SerializeField] private int fps = 60;
        [SerializeField] private bool autoStartRecording;
        [SerializeField] private bool flipVertical = true;
        [SerializeField] private string outputRootName = "record";
        [SerializeField] private string leftVideoFileName = "left_recording.mp4";
        [SerializeField] private string rightVideoFileName = "right_recording.mp4";
        [SerializeField] private string metadataFileName = "quest_camera_metadata.json";
        [SerializeField] private string leftFrameMetadataFileName = "left_frames.jsonl";
        [SerializeField] private string rightFrameMetadataFileName = "right_frames.jsonl";
        [SerializeField] private string trajectoryFileName = "trajectory.jsonl";

        [Header("Synchronized Trajectory")]
        [SerializeField] private bool recordSynchronizedTrajectory = true;
        [SerializeField] private PaperTrackerOscReceiver gazeReceiver;
        [SerializeField] private EyeGazePose eyeGazePose;
        [SerializeField] private bool allowHeadCenterGazeFallback = true;
        [SerializeField] private float gazeRayOffset = 0.1f;
        [SerializeField] private float gazeMaxDistance = 10f;

        [Header("Realtime Telemetry")]
        [SerializeField] private QuestRecordingTelemetrySender telemetrySender;

        [Header("HMD Eye Pose Telemetry")]
        [SerializeField] private bool recordHmdEyePoses = true;

        private Mp4RecorderPure leftRecorder;
        private Mp4RecorderPure rightRecorder;
        private Texture leftTexture;
        private Texture rightTexture;
        private long leftTimestampPrev;
        private long rightTimestampPrev;
        private string outputDirectory;
        private bool isReady;
        private bool isRecording;
        private string lastInitStatus = "not_started";
        private string lastInitError;
        private string leftTextureType;
        private string rightTextureType;
        private int leftTextureWidth;
        private int leftTextureHeight;
        private int rightTextureWidth;
        private int rightTextureHeight;
        private DateTime startTimeUtc;
        private double recordingStartRealtimeSeconds;
        private int trajectorySampleIndex;
        private int liveSampleIndex;
        private int leftDeltaCount;
        private int rightDeltaCount;
        private FrameMetadata latestLeftFrame;
        private FrameMetadata latestRightFrame;
        private bool hasLatestLeftFrame;
        private bool hasLatestRightFrame;
        private StreamWriter leftFrameWriter;
        private StreamWriter rightFrameWriter;
        private StreamWriter trajectoryWriter;
        private readonly RecordMetadata metadata = new();
        private OVRCameraRig cachedCameraRig;

        public bool IsReady => isReady;
        public bool IsRecording => isRecording;
        public string OutputDirectory => outputDirectory;
        public bool RecordRightCamera => recordRightCamera;
        public bool RecordSynchronizedTrajectory => recordSynchronizedTrajectory;
        public bool HasLeftCameraAccess => leftCameraAccess != null;
        public bool HasRightCameraAccess => rightCameraAccess != null;
        public bool HasGazeReceiver => gazeReceiver != null;
        public bool GazeReceiverConnected => gazeReceiver != null && gazeReceiver.IsConnected;
        public string LastInitStatus => lastInitStatus;
        public string LastInitError => lastInitError;
        public bool LeftCameraIsPlaying => leftCameraAccess != null && leftCameraAccess.IsPlaying;
        public bool RightCameraIsPlaying => rightCameraAccess != null && rightCameraAccess.IsPlaying;
        public string LeftTextureType => leftTextureType;
        public string RightTextureType => rightTextureType;
        public int LeftTextureWidth => leftTextureWidth;
        public int LeftTextureHeight => leftTextureHeight;
        public int RightTextureWidth => rightTextureWidth;
        public int RightTextureHeight => rightTextureHeight;

        private IEnumerator Start()
        {
            lastInitStatus = "requesting_permissions";
            lastInitError = null;
            RequestCameraPermissions();
            bool leftReady = false;
            lastInitStatus = "waiting_left_camera";
            yield return WaitForCamera(leftCameraAccess, "left", ready => leftReady = ready);
            if (!leftReady)
            {
                FailInitialization("left_camera_not_ready");
                yield break;
            }

            leftTexture = leftCameraAccess.GetTexture();
            CaptureTextureStatus(leftTexture, true);
            if (leftTexture == null)
            {
                FailInitialization("left_texture_null");
                yield break;
            }

            leftRecorder = new Mp4RecorderPure(0, leftTexture.width, leftTexture.height, fps, leftVideoFileName, flipVertical);
            leftTimestampPrev = CameraTimestampMs(leftCameraAccess);

            if (rightCameraAccess != null)
            {
                bool rightReady = false;
                lastInitStatus = "waiting_right_camera";
                yield return WaitForCamera(rightCameraAccess, "right", ready => rightReady = ready);
                if (!rightReady)
                {
                    FailInitialization("right_camera_not_ready");
                    yield break;
                }

                rightTimestampPrev = CameraTimestampMs(rightCameraAccess);
            }

            if (recordRightCamera)
            {
                if (rightCameraAccess == null)
                {
                    FailInitialization("missing_right_camera_reference");
                    yield break;
                }

                rightTexture = rightCameraAccess.GetTexture();
                CaptureTextureStatus(rightTexture, false);
                if (rightTexture == null)
                {
                    FailInitialization("right_texture_null");
                    yield break;
                }

                rightRecorder = new Mp4RecorderPure(1, rightTexture.width, rightTexture.height, fps, rightVideoFileName, flipVertical);
            }

            isReady = true;
            lastInitStatus = "ready";
            lastInitError = null;
            Debug.Log($"[QuestCameraRecorder] Ready. left={leftTexture.width}x{leftTexture.height} type={leftTexture.GetType().Name}, fps={fps}");

            if (autoStartRecording)
            {
                StartRecording();
            }
        }

        private void FailInitialization(string reason)
        {
            isReady = false;
            lastInitStatus = "failed";
            lastInitError = reason;
            Debug.LogError("[QuestCameraRecorder] Initialization failed: " + reason, this);
        }

        private void CaptureTextureStatus(Texture texture, bool left)
        {
            string type = texture != null ? texture.GetType().Name : "null";
            int textureWidth = texture != null ? texture.width : 0;
            int textureHeight = texture != null ? texture.height : 0;
            if (left)
            {
                leftTextureType = type;
                leftTextureWidth = textureWidth;
                leftTextureHeight = textureHeight;
            }
            else
            {
                rightTextureType = type;
                rightTextureWidth = textureWidth;
                rightTextureHeight = textureHeight;
            }
        }

        private void Update()
        {
            if (isRecording)
            {
                RecordCameraIfNewFrame(
                    leftCameraAccess,
                    leftTexture,
                    leftRecorder,
                    ref leftTimestampPrev,
                    leftFrameWriter,
                    ref latestLeftFrame,
                    ref hasLatestLeftFrame,
                    ref metadata.leftFrameCount);

                if (recordRightCamera)
                {
                    RecordCameraIfNewFrame(
                        rightCameraAccess,
                        rightTexture,
                        rightRecorder,
                        ref rightTimestampPrev,
                        rightFrameWriter,
                        ref latestRightFrame,
                        ref hasLatestRightFrame,
                        ref metadata.rightFrameCount);
                }

                if (recordSynchronizedTrajectory)
                {
                    RecordTrajectorySample();
                }
            }
            else
            {
                SendLiveTelemetrySample();
            }
        }

        public void StartRecording()
        {
            if (!isReady)
            {
                Debug.LogWarning("[QuestCameraRecorder] Cannot start: recorder is not ready yet.");
                return;
            }

            if (isRecording)
            {
                return;
            }

            startTimeUtc = DateTime.UtcNow;
            outputDirectory = Path.Combine(
                Application.persistentDataPath,
                outputRootName,
                $"record_{DateTime.Now:yyyyMMdd_HHmmss}");
            Directory.CreateDirectory(outputDirectory);

            metadata.Reset();
            metadata.schemaVersion = "quest_camera_recorder_metadata_v6";
            metadata.startTimeUtc = startTimeUtc.ToString("o", CultureInfo.InvariantCulture);
            metadata.outputDirectory = outputDirectory;
            metadata.fps = fps;
            metadata.leftVideoFileName = leftVideoFileName;
            metadata.rightVideoFileName = recordRightCamera ? rightVideoFileName : null;
            metadata.leftFrameMetadataFileName = leftFrameMetadataFileName;
            metadata.rightFrameMetadataFileName = recordRightCamera ? rightFrameMetadataFileName : null;
            metadata.trajectoryFileName = recordSynchronizedTrajectory ? trajectoryFileName : null;
            metadata.hasLeftCameraAccess = leftCameraAccess != null;
            metadata.hasRightCameraAccess = rightCameraAccess != null;
            metadata.recordRightCamera = recordRightCamera;
            metadata.leftIntrinsics = IntrinsicsToArray(leftCameraAccess);
            metadata.rightIntrinsics = rightCameraAccess != null ? IntrinsicsToArray(rightCameraAccess) : null;
            metadata.leftResolution = ResolutionToArray(leftCameraAccess);
            metadata.rightResolution = rightCameraAccess != null ? ResolutionToArray(rightCameraAccess) : null;
            metadata.trajectoryFrame = "unity_world";
            metadata.trajectorySampleSource = "Update loop, one sample per rendered frame while recording";
            metadata.trajectoryContents = "left/right passthrough camera pose, HMD left/right eye pose, nearest left/right video frame index and time delta, PaperTracker gaze ray, optional EyeGazePose pitch/yaw bias, gazePoint3DWorld from EyeGazePose.vizObj Transform.position when visible, optional EnvironmentMapper gazePointWorld depth hit for diagnostics, and left/right controller pose in Unity world frame when available";
            metadata.cameraPoseConvention = "PassthroughCameraAccess.GetCameraPose in Unity world frame";
            metadata.gazePointConvention = "gazePoint3DWorld is EyeGazePose.vizObj Transform.position in Unity world frame; gazePointWorld is the EnvironmentMapper depth hit when available; gazePoint3DSource records the visual-object source.";
            metadata.gazeProjectionConvention = "left/rightGazePointViewport are normalized camera viewport coordinates with origin bottom-left. left/rightGazePointPixel are recorded-video pixel coordinates with origin top-left; recorder flipVertical is already applied. Projection uses passthrough camera intrinsics as a pinhole camera with no distortion.";
            metadata.cameraDistortionModel = "pinhole_no_distortion";
            metadata.trajectorySyncConvention = "nearestLeftFrameIndex/nearestRightFrameIndex refer to the most recently recorded video frame when the trajectory sample is written; frameTimeDeltaSeconds is trajectory unity time minus that frame unity time";
            metadata.realtimeTelemetryProtocol = QuestRecordingTelemetrySender.ProtocolName;
            metadata.realtimeTelemetryContents = "UDP JSON lines: recording_start, sample, recording_stop. sample includes gazePoint3DWorld, HMD left/right eye pose, and left/right controller poses in Unity world frame.";
            metadata.recordSynchronizedTrajectory = recordSynchronizedTrajectory;
            metadata.recordHmdEyePoses = recordHmdEyePoses;
            if (gazeReceiver == null)
            {
                gazeReceiver = FindFirstObjectByType<PaperTrackerOscReceiver>();
            }

            ResolveTelemetrySender();
            metadata.realtimeTelemetryEnabled = telemetrySender != null && telemetrySender.SendTelemetry;
            metadata.realtimeTelemetryHost = telemetrySender != null ? telemetrySender.Host : null;
            metadata.realtimeTelemetryPort = telemetrySender != null ? telemetrySender.Port : 0;

            metadata.gazeSource = gazeReceiver != null ? "PaperTrackerOscReceiver" : null;
            metadata.allowHeadCenterGazeFallback = allowHeadCenterGazeFallback;
            if (!recordSynchronizedTrajectory)
            {
                Debug.LogWarning("[QuestCameraRecorder] Synchronized trajectory recording is disabled; metadata will not include per-frame camera/gaze trajectory samples.", this);
            }

            OpenJsonlWriters();
            leftRecorder.StartRecording(outputDirectory);
            rightRecorder?.StartRecording(outputDirectory);
            recordingStartRealtimeSeconds = Time.realtimeSinceStartupAsDouble;
            trajectorySampleIndex = 0;
            leftDeltaCount = 0;
            rightDeltaCount = 0;
            hasLatestLeftFrame = false;
            hasLatestRightFrame = false;

            isRecording = true;
            telemetrySender?.NotifyRecordingStarted(outputDirectory, Path.GetFileName(outputDirectory), startTimeUtc);
            Debug.Log("[QuestCameraRecorder] Recording started: " + outputDirectory);
        }

        public void ToggleRecording()
        {
            if (isRecording)
            {
                StopRecording();
            }
            else
            {
                StartRecording();
            }
        }

        public void StopRecording()
        {
            if (!isRecording)
            {
                return;
            }

            isRecording = false;
            telemetrySender?.NotifyRecordingStopped();
            leftRecorder?.StopRecording();
            rightRecorder?.StopRecording();
            CloseJsonlWriters();

            metadata.durationSeconds = (DateTime.UtcNow - startTimeUtc).TotalSeconds;
            metadata.leftVideoSha256 = leftRecorder?.FinalSha256;
            metadata.rightVideoSha256 = rightRecorder?.FinalSha256;

            if (leftDeltaCount > 0)
            {
                metadata.meanAbsLeftFrameTimeDeltaSeconds /= leftDeltaCount;
            }

            if (rightDeltaCount > 0)
            {
                metadata.meanAbsRightFrameTimeDeltaSeconds /= rightDeltaCount;
            }

            metadata.gazeSampleRatio = Ratio(metadata.gazeSampleCount, metadata.trajectorySampleCount);
            metadata.gazeHitSampleRatio = Ratio(metadata.gazeHitSampleCount, metadata.trajectorySampleCount);
            metadata.leftFrameTrajectoryCoverageRatio = Ratio(leftDeltaCount, metadata.trajectorySampleCount);
            metadata.rightFrameTrajectoryCoverageRatio = Ratio(rightDeltaCount, metadata.trajectorySampleCount);
            metadata.recordingQualitySummary = BuildRecordingQualitySummary();
            SaveMetadata();

            Debug.Log(
                $"[QuestCameraRecorder] Recording stopped: {outputDirectory}. " +
                $"leftFrames={metadata.leftFrameCount} rightFrames={metadata.rightFrameCount} " +
                $"trajectorySamples={metadata.trajectorySampleCount} gazeSamples={metadata.gazeSampleCount} gazeHits={metadata.gazeHitSampleCount} " +
                $"quality={metadata.recordingQualitySummary} gazeHitRatio={metadata.gazeHitSampleRatio:0.000} " +
                $"leftMeanDt={metadata.meanAbsLeftFrameTimeDeltaSeconds:0.0000}s rightMeanDt={metadata.meanAbsRightFrameTimeDeltaSeconds:0.0000}s");
        }

        private void OpenJsonlWriters()
        {
            CloseJsonlWriters();
            leftFrameWriter = CreateJsonlWriter(Path.Combine(outputDirectory, leftFrameMetadataFileName));
            if (recordRightCamera)
            {
                rightFrameWriter = CreateJsonlWriter(Path.Combine(outputDirectory, rightFrameMetadataFileName));
            }

            if (recordSynchronizedTrajectory)
            {
                trajectoryWriter = CreateJsonlWriter(Path.Combine(outputDirectory, trajectoryFileName));
            }
        }

        private void CloseJsonlWriters()
        {
            CloseWriter(ref leftFrameWriter);
            CloseWriter(ref rightFrameWriter);
            CloseWriter(ref trajectoryWriter);
        }

        private static StreamWriter CreateJsonlWriter(string path)
        {
            return new StreamWriter(
                new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 65536),
                new UTF8Encoding(false));
        }

        private static void CloseWriter(ref StreamWriter writer)
        {
            if (writer == null)
            {
                return;
            }

            writer.Flush();
            writer.Dispose();
            writer = null;
        }

        private string BuildRecordingQualitySummary()
        {
            if (metadata.trajectorySampleCount <= 0)
            {
                return "no_trajectory_samples";
            }

            if (recordRightCamera && metadata.rightFrameCount <= 0)
            {
                return "missing_right_video_frames";
            }

            if (metadata.leftFrameCount <= 0)
            {
                return "missing_left_video_frames";
            }

            if (metadata.gazeSampleCount <= 0)
            {
                return "missing_gaze_samples";
            }

            if (metadata.gazeHitSampleCount <= 0)
            {
                return "missing_3d_gaze_hits";
            }

            return "ok";
        }

        private static void RequestCameraPermissions()
        {
            OVRPermissionsRequester.Request(new[]
            {
                OVRPermissionsRequester.Permission.Scene,
                OVRPermissionsRequester.Permission.PassthroughCameraAccess
            });
        }

        private static IEnumerator WaitForCamera(
            PassthroughCameraAccess cameraAccess,
            string label,
            Action<bool> onComplete)
        {
            if (cameraAccess == null)
            {
                Debug.LogError($"[QuestCameraRecorder] Missing {label} PassthroughCameraAccess reference.");
                onComplete?.Invoke(false);
                yield break;
            }

            while (!cameraAccess.IsPlaying)
            {
                yield return null;
            }

            onComplete?.Invoke(true);
        }

        private void RecordTrajectorySample()
        {
            TrajectorySample sample = BuildTrajectorySample(
                trajectorySampleIndex++,
                -1,
                now => now - recordingStartRealtimeSeconds,
                true,
                "recording");

            metadata.trajectorySampleCount++;
            AccumulateTrajectoryStats(sample);
            WriteJsonl(trajectoryWriter, sample);
            telemetrySender?.SendSample(sample);
        }

        public TrajectorySample CaptureExternalTrajectorySample(
            int externalSampleIndex,
            int nearestLeftFrameIndex,
            int nearestRightFrameIndex,
            double recordingTimestampSeconds,
            string telemetryMode)
        {
            bool previousHasLeftFrame = hasLatestLeftFrame;
            bool previousHasRightFrame = hasLatestRightFrame;
            FrameMetadata previousLeftFrame = latestLeftFrame;
            FrameMetadata previousRightFrame = latestRightFrame;

            double now = Time.realtimeSinceStartupAsDouble;
            if (nearestLeftFrameIndex >= 0)
            {
                latestLeftFrame = new FrameMetadata
                {
                    frameIndex = nearestLeftFrameIndex,
                    cameraTimestampSeconds = leftCameraAccess != null ? CameraTimestampMs(leftCameraAccess) / 1000.0 : -1.0,
                    unityTimestampSeconds = now,
                    pose = leftCameraAccess != null ? PoseToArray(leftCameraAccess.GetCameraPose()) : null
                };
                hasLatestLeftFrame = true;
            }

            if (nearestRightFrameIndex >= 0)
            {
                latestRightFrame = new FrameMetadata
                {
                    frameIndex = nearestRightFrameIndex,
                    cameraTimestampSeconds = rightCameraAccess != null ? CameraTimestampMs(rightCameraAccess) / 1000.0 : -1.0,
                    unityTimestampSeconds = now,
                    pose = rightCameraAccess != null ? PoseToArray(rightCameraAccess.GetCameraPose()) : null
                };
                hasLatestRightFrame = true;
            }

            TrajectorySample sample = BuildTrajectorySample(
                externalSampleIndex,
                -1,
                _ => recordingTimestampSeconds,
                true,
                telemetryMode);

            latestLeftFrame = previousLeftFrame;
            latestRightFrame = previousRightFrame;
            hasLatestLeftFrame = previousHasLeftFrame;
            hasLatestRightFrame = previousHasRightFrame;
            return sample;
        }

        private void SendLiveTelemetrySample()
        {
            if (!isReady)
            {
                return;
            }

            ResolveTelemetrySender();
            if (telemetrySender == null || !telemetrySender.CanSendSampleNow())
            {
                return;
            }

            TrajectorySample sample = BuildTrajectorySample(
                -1,
                liveSampleIndex++,
                _ => 0.0,
                false,
                "live_preview");
            telemetrySender.SendSample(sample);
        }

        private TrajectorySample BuildTrajectorySample(
            int sampleIndex,
            int liveIndex,
            Func<double, double> recordingTimeSeconds,
            bool sampleIsRecording,
            string telemetryMode)
        {
            if (gazeReceiver == null)
            {
                gazeReceiver = FindFirstObjectByType<PaperTrackerOscReceiver>();
            }

            if (eyeGazePose == null)
            {
                eyeGazePose = FindFirstObjectByType<EyeGazePose>();
            }

            bool hasLeftPose = leftCameraAccess != null;
            Pose leftPose = hasLeftPose ? leftCameraAccess.GetCameraPose() : default;
            bool hasRightPose = rightCameraAccess != null;
            Pose rightPose = hasRightPose ? rightCameraAccess.GetCameraPose() : default;
            Pose leftEyePose = default;
            Pose rightEyePose = default;
            string leftEyePoseSource = null;
            string rightEyePoseSource = null;
            bool hasLeftEyePose = recordHmdEyePoses &&
                TryGetHmdEyePose(UnityEngine.XR.XRNode.LeftEye, out leftEyePose, out leftEyePoseSource);
            bool hasRightEyePose = recordHmdEyePoses &&
                TryGetHmdEyePose(UnityEngine.XR.XRNode.RightEye, out rightEyePose, out rightEyePoseSource);
            double now = Time.realtimeSinceStartupAsDouble;

            int nearestLeftFrameIndex = hasLatestLeftFrame ? latestLeftFrame.frameIndex : -1;
            int nearestRightFrameIndex = hasLatestRightFrame ? latestRightFrame.frameIndex : -1;
            TrajectorySample sample = new TrajectorySample
            {
                sampleIndex = sampleIndex,
                liveSampleIndex = liveIndex,
                isRecording = sampleIsRecording,
                telemetryMode = telemetryMode,
                unityTimestampSeconds = now,
                recordingTimestampSeconds = recordingTimeSeconds(now),
                hasLeftCameraPose = hasLeftPose,
                hasRightCameraPose = hasRightPose,
                nearestLeftFrameIndex = nearestLeftFrameIndex,
                nearestRightFrameIndex = nearestRightFrameIndex,
                leftCameraTimestampSeconds = hasLeftPose ? CameraTimestampMs(leftCameraAccess) / 1000.0 : -1.0,
                rightCameraTimestampSeconds = hasRightPose ? CameraTimestampMs(rightCameraAccess) / 1000.0 : -1.0,
                leftCameraPose = hasLeftPose ? PoseToArray(leftPose) : null,
                rightCameraPose = hasRightPose ? PoseToArray(rightPose) : null,
                leftPassthroughCameraPose = hasLeftPose ? PoseToArray(leftPose) : null,
                rightPassthroughCameraPose = hasRightPose ? PoseToArray(rightPose) : null,
                hasLeftEyePose = hasLeftEyePose,
                hasRightEyePose = hasRightEyePose,
                leftEyePoseSource = hasLeftEyePose ? leftEyePoseSource : null,
                rightEyePoseSource = hasRightEyePose ? rightEyePoseSource : null,
                leftEyePose = hasLeftEyePose ? PoseToArray(leftEyePose) : null,
                rightEyePose = hasRightEyePose ? PoseToArray(rightEyePose) : null,
                leftEyePosition = hasLeftEyePose ? Vector3ToArray(leftEyePose.position) : null,
                rightEyePosition = hasRightEyePose ? Vector3ToArray(rightEyePose.position) : null,
                leftController = telemetrySender != null
                    ? telemetrySender.CaptureControllerTelemetry(UnityEngine.XR.XRNode.LeftHand)
                    : null,
                rightController = telemetrySender != null
                    ? telemetrySender.CaptureControllerTelemetry(UnityEngine.XR.XRNode.RightHand)
                    : null
            };
            sample.nearestLeftFrameTimeDeltaSeconds = FrameTimeDeltaSeconds(latestLeftFrame, hasLatestLeftFrame, now);
            sample.nearestRightFrameTimeDeltaSeconds = FrameTimeDeltaSeconds(latestRightFrame, hasLatestRightFrame, now);

            if (TryGetGazeRay(out Ray gazeRay, out bool usingBiasedEyeGazePose))
            {
                sample.hasGaze = true;
                sample.gazeSource = usingBiasedEyeGazePose
                    ? "EyeGazePoseBiasedPaperTrackerOscReceiver"
                    : gazeReceiver != null && gazeReceiver.IsConnected
                        ? "PaperTrackerOscReceiver"
                        : "EyeGazePoseScreenPointFallback";
                sample.gazePitchYaw = gazeReceiver != null ? Vector2ToArray(gazeReceiver.CenterPitchYaw) : null;
                sample.gazePitchYawBias = usingBiasedEyeGazePose ? Vector2ToArray(eyeGazePose.OscPitchYawBiasDegrees) : null;
                sample.gazeBiasedPitchYaw = usingBiasedEyeGazePose ? Vector2ToArray(eyeGazePose.LatestBiasedPitchYaw) : sample.gazePitchYaw;
                sample.gazeRayOrigin = Vector3ToArray(gazeRay.origin);
                sample.gazeRayDirection = Vector3ToArray(gazeRay.direction);
                Vector3 fallbackPoint = gazeRay.GetPoint(gazeMaxDistance);
                sample.gazeFallbackPointWorld = Vector3ToArray(fallbackPoint);

                if (TryRaycastGazePoint(gazeRay, out EnvironmentMapper.RayResult hit))
                {
                    sample.hasGazeHit = true;
                    sample.gazePointWorld = Vector3ToArray(hit.point);
                }
                else
                {
                    sample.gazeHitMissReason = LastGazeHitMissReason();
                }

                if (TryGetEyeGazeVizPoint(out Vector3 vizPoint))
                {
                    sample.gazePoint3DWorld = Vector3ToArray(vizPoint);
                    sample.gazePoint3DSource = "eyegazepose_vizobj_transform";
                }
                else
                {
                    sample.gazePoint3DSource = "missing_eyegazepose_vizobj_transform";
                }

                RecordGazePointProjections(sample, leftPose, rightPose);
            }
            else
            {
                sample.gazeMissReason = gazeReceiver == null
                    ? "missing_gaze_receiver"
                    : "gaze_receiver_not_connected";
            }

            return sample;
        }

        private void ResolveTelemetrySender()
        {
            if (telemetrySender != null)
            {
                return;
            }

            telemetrySender = GetComponent<QuestRecordingTelemetrySender>();
            if (telemetrySender != null)
            {
                return;
            }

            telemetrySender = FindFirstObjectByType<QuestRecordingTelemetrySender>();
            if (telemetrySender == null)
            {
                telemetrySender = gameObject.AddComponent<QuestRecordingTelemetrySender>();
            }
        }

        private void AccumulateTrajectoryStats(TrajectorySample sample)
        {
            if (sample == null)
            {
                return;
            }

            if (sample.hasGaze)
            {
                metadata.gazeSampleCount++;
            }

            if (sample.hasGazeHit)
            {
                metadata.gazeHitSampleCount++;
            }

            if (!sample.hasGazeHit && !string.IsNullOrEmpty(sample.gazeHitMissReason))
            {
                metadata.gazeHitMissSampleCount++;
            }

            AccumulateAbsDelta(
                sample.nearestLeftFrameTimeDeltaSeconds,
                ref metadata.meanAbsLeftFrameTimeDeltaSeconds,
                ref metadata.maxAbsLeftFrameTimeDeltaSeconds,
                ref leftDeltaCount);
            AccumulateAbsDelta(
                sample.nearestRightFrameTimeDeltaSeconds,
                ref metadata.meanAbsRightFrameTimeDeltaSeconds,
                ref metadata.maxAbsRightFrameTimeDeltaSeconds,
                ref rightDeltaCount);
        }

        private void RecordGazePointProjections(TrajectorySample sample, Pose leftPose, Pose rightPose)
        {
            if (sample == null || sample.gazePoint3DWorld == null || sample.gazePoint3DWorld.Length != 3)
            {
                return;
            }

            Vector3 gazePoint = new Vector3(
                (float)sample.gazePoint3DWorld[0],
                (float)sample.gazePoint3DWorld[1],
                (float)sample.gazePoint3DWorld[2]);

            if (TryProjectWorldPoint(leftCameraAccess, leftPose, gazePoint, flipVertical, out CameraProjection leftProjection))
            {
                sample.hasLeftGazePointProjection = true;
                sample.leftGazePointViewport = Vector2ToArray(leftProjection.viewport);
                sample.leftGazePointPixel = Vector2ToArray(leftProjection.pixel);
                sample.leftGazePointInImage = leftProjection.inImage;
                sample.leftGazePointCameraZ = leftProjection.cameraZ;
                sample.leftGazePointProjectionSource = sample.gazePoint3DSource;
            }
            else
            {
                sample.leftGazePointProjectionMissReason = ProjectionMissReason(leftCameraAccess, leftPose, gazePoint);
            }

            if (TryProjectWorldPoint(rightCameraAccess, rightPose, gazePoint, flipVertical, out CameraProjection rightProjection))
            {
                sample.hasRightGazePointProjection = true;
                sample.rightGazePointViewport = Vector2ToArray(rightProjection.viewport);
                sample.rightGazePointPixel = Vector2ToArray(rightProjection.pixel);
                sample.rightGazePointInImage = rightProjection.inImage;
                sample.rightGazePointCameraZ = rightProjection.cameraZ;
                sample.rightGazePointProjectionSource = sample.gazePoint3DSource;
            }
            else
            {
                sample.rightGazePointProjectionMissReason = ProjectionMissReason(rightCameraAccess, rightPose, gazePoint);
            }
        }

        private static bool TryProjectWorldPoint(
            PassthroughCameraAccess cameraAccess,
            Pose cameraPose,
            Vector3 worldPoint,
            bool outputVideoIsFlippedVertically,
            out CameraProjection projection)
        {
            projection = default;
            if (cameraAccess == null || !cameraAccess.IsPlaying)
            {
                return false;
            }

            Vector2Int resolution = cameraAccess.CurrentResolution;
            if (resolution.x <= 0 || resolution.y <= 0)
            {
                return false;
            }

            PassthroughCameraAccess.CameraIntrinsics intrinsics = cameraAccess.Intrinsics;
            if (intrinsics.FocalLength.x == 0f || intrinsics.FocalLength.y == 0f ||
                intrinsics.SensorResolution.x <= 0 || intrinsics.SensorResolution.y <= 0)
            {
                return false;
            }

            Vector3 cameraPoint = Quaternion.Inverse(cameraPose.rotation) * (worldPoint - cameraPose.position);
            if (cameraPoint.z <= 0f)
            {
                return false;
            }

            Rect crop = SensorCropRegion(intrinsics.SensorResolution, resolution);
            Vector2 sensorPoint = new Vector2(
                cameraPoint.x / cameraPoint.z * intrinsics.FocalLength.x + intrinsics.PrincipalPoint.x,
                cameraPoint.y / cameraPoint.z * intrinsics.FocalLength.y + intrinsics.PrincipalPoint.y);
            Vector2 viewport = new Vector2(
                (sensorPoint.x - crop.x) / crop.width,
                (sensorPoint.y - crop.y) / crop.height);

            float videoY = outputVideoIsFlippedVertically ? 1f - viewport.y : viewport.y;
            Vector2 pixel = new Vector2(
                viewport.x * resolution.x,
                videoY * resolution.y);

            projection = new CameraProjection
            {
                viewport = viewport,
                pixel = pixel,
                inImage = viewport.x >= 0f && viewport.x <= 1f && viewport.y >= 0f && viewport.y <= 1f,
                cameraZ = cameraPoint.z
            };
            return true;
        }

        private static Rect SensorCropRegion(Vector2Int sensorResolutionInt, Vector2Int currentResolutionInt)
        {
            Vector2 sensorResolution = sensorResolutionInt;
            Vector2 currentResolution = currentResolutionInt;
            Vector2 scaleFactor = currentResolution / sensorResolution;
            scaleFactor /= Mathf.Max(scaleFactor.x, scaleFactor.y);
            return new Rect(
                sensorResolution.x * (1f - scaleFactor.x) * 0.5f,
                sensorResolution.y * (1f - scaleFactor.y) * 0.5f,
                sensorResolution.x * scaleFactor.x,
                sensorResolution.y * scaleFactor.y);
        }

        private static string ProjectionMissReason(PassthroughCameraAccess cameraAccess, Pose cameraPose, Vector3 worldPoint)
        {
            if (cameraAccess == null)
            {
                return "missing_camera_access";
            }

            if (!cameraAccess.IsPlaying)
            {
                return "camera_not_playing";
            }

            Vector2Int resolution = cameraAccess.CurrentResolution;
            if (resolution.x <= 0 || resolution.y <= 0)
            {
                return "invalid_camera_resolution";
            }

            PassthroughCameraAccess.CameraIntrinsics intrinsics = cameraAccess.Intrinsics;
            if (intrinsics.FocalLength.x == 0f || intrinsics.FocalLength.y == 0f ||
                intrinsics.SensorResolution.x <= 0 || intrinsics.SensorResolution.y <= 0)
            {
                return "invalid_camera_intrinsics";
            }

            Vector3 cameraPoint = Quaternion.Inverse(cameraPose.rotation) * (worldPoint - cameraPose.position);
            if (cameraPoint.z <= 0f)
            {
                return "point_behind_camera";
            }

            return "unknown_projection_failure";
        }

        private bool TryRaycastGazePoint(Ray gazeRay, out EnvironmentMapper.RayResult hit)
        {
            hit = default;
            if (EnvironmentMapper.Instance == null || gazeMaxDistance <= 0f)
            {
                return false;
            }

            try
            {
                return EnvironmentMapper.Raycast(
                    gazeRay,
                    gazeMaxDistance,
                    out hit,
                    EnvironmentMapper.RaycastMode.Negative);
            }
            catch (Exception exception)
            {
                Debug.LogWarning("[QuestCameraRecorder] Gaze depth raycast failed; no EnvironmentMapper diagnostic point will be written. " + exception.Message, this);
                return false;
            }
        }

        private string LastGazeHitMissReason()
        {
            if (EnvironmentMapper.Instance == null)
            {
                return "missing_environment_mapper";
            }

            if (gazeMaxDistance <= 0f)
            {
                return "invalid_gaze_max_distance";
            }

            return "no_environment_depth_hit";
        }

        private bool TryGetEyeGazeVizPoint(out Vector3 point)
        {
            point = default;
            if (eyeGazePose == null)
            {
                eyeGazePose = FindFirstObjectByType<EyeGazePose>();
            }

            if (eyeGazePose == null || !eyeGazePose.HasVizPoint)
            {
                return false;
            }

            point = eyeGazePose.VizPointWorld;
            return IsFinite(point);
        }

        private static bool IsFinite(Vector3 value)
        {
            return !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
                   !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
                   !float.IsNaN(value.z) && !float.IsInfinity(value.z);
        }

        private bool TryGetGazeRay(out Ray ray, out bool usingBiasedEyeGazePose)
        {
            usingBiasedEyeGazePose = false;

            if (eyeGazePose == null)
            {
                eyeGazePose = FindFirstObjectByType<EyeGazePose>();
            }

            if (eyeGazePose != null && eyeGazePose.TryGetBiasedOscRay(out ray))
            {
                usingBiasedEyeGazePose = true;
                return true;
            }

            if (gazeReceiver != null && gazeReceiver.IsConnected)
            {
                ray = gazeReceiver.WorldGazeRay;
                ray.origin += ray.direction * gazeRayOffset;
                return true;
            }

            if (allowHeadCenterGazeFallback && eyeGazePose != null && eyeGazePose.TryGetFallbackScreenPointRay(out ray))
            {
                return true;
            }

            ray = default;
            return false;
        }

        private bool TryGetHmdEyePose(UnityEngine.XR.XRNode node, out Pose pose, out string source)
        {
            if (cachedCameraRig == null)
            {
                cachedCameraRig = FindFirstObjectByType<OVRCameraRig>();
            }

            if (cachedCameraRig != null)
            {
                Transform anchor = node == UnityEngine.XR.XRNode.LeftEye
                    ? cachedCameraRig.leftEyeAnchor
                    : cachedCameraRig.rightEyeAnchor;
                if (anchor != null && anchor.gameObject.activeInHierarchy)
                {
                    pose = new Pose(anchor.position, anchor.rotation);
                    source = node == UnityEngine.XR.XRNode.LeftEye
                        ? "OVRCameraRig.leftEyeAnchor"
                        : "OVRCameraRig.rightEyeAnchor";
                    return true;
                }
            }

            UnityEngine.XR.InputDevice device = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(node);
            Vector3 position = Vector3.zero;
            Quaternion rotation = Quaternion.identity;
            bool positionTracked = device.isValid &&
                device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.devicePosition, out position);
            bool rotationTracked = device.isValid &&
                device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.deviceRotation, out rotation);
            if (positionTracked || rotationTracked)
            {
                pose = new Pose(
                    positionTracked ? position : Vector3.zero,
                    rotationTracked ? rotation : Quaternion.identity);
                source = node == UnityEngine.XR.XRNode.LeftEye
                    ? "XRInputDevice.LeftEye"
                    : "XRInputDevice.RightEye";
                return true;
            }

            pose = default;
            source = null;
            return false;
        }

        private static void RecordCameraIfNewFrame(
            PassthroughCameraAccess cameraAccess,
            Texture texture,
            Mp4RecorderPure recorder,
            ref long previousTimestampMs,
            StreamWriter frameWriter,
            ref FrameMetadata latestFrame,
            ref bool hasLatestFrame,
            ref int frameCount)
        {
            if (cameraAccess == null || texture == null || recorder == null)
            {
                return;
            }

            long timestampMs = CameraTimestampMs(cameraAccess);
            if (timestampMs == previousTimestampMs)
            {
                return;
            }

            previousTimestampMs = timestampMs;
            recorder.RecordFrame(texture);

            Pose pose = cameraAccess.GetCameraPose();
            latestFrame = new FrameMetadata
            {
                frameIndex = frameCount,
                cameraTimestampSeconds = timestampMs / 1000.0,
                unityTimestampSeconds = Time.realtimeSinceStartupAsDouble,
                pose = PoseToArray(pose)
            };
            hasLatestFrame = true;
            frameCount++;
            WriteJsonl(frameWriter, latestFrame);
        }

        private static double FrameTimeDeltaSeconds(FrameMetadata frame, bool hasFrame, double sampleUnityTimeSeconds)
        {
            if (!hasFrame)
            {
                return -1.0;
            }

            return sampleUnityTimeSeconds - frame.unityTimestampSeconds;
        }

        private static void WriteJsonl(StreamWriter writer, object value)
        {
            if (writer == null || value == null)
            {
                return;
            }

            writer.WriteLine(JsonUtility.ToJson(value, false));
        }

        private static void AccumulateAbsDelta(double deltaSeconds, ref double sumAbsDelta, ref double maxAbsDelta, ref int count)
        {
            if (deltaSeconds < 0.0)
            {
                return;
            }

            double absDelta = Math.Abs(deltaSeconds);
            sumAbsDelta += absDelta;
            maxAbsDelta = Math.Max(maxAbsDelta, absDelta);
            count++;
        }

        private static double Ratio(int numerator, int denominator)
        {
            return denominator > 0 ? (double)numerator / denominator : 0.0;
        }

        private void SaveMetadata()
        {
            if (string.IsNullOrEmpty(outputDirectory))
            {
                return;
            }

            string metadataPath = Path.Combine(outputDirectory, metadataFileName);
            File.WriteAllText(metadataPath, metadata.ToJson(), new UTF8Encoding(false));
        }

        private static long CameraTimestampMs(PassthroughCameraAccess cameraAccess)
        {
            return new DateTimeOffset(cameraAccess.Timestamp).ToUnixTimeMilliseconds();
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

        private static double[] Vector3ToArray(Vector3 value)
        {
            return new[] { (double)value.x, value.y, value.z };
        }

        private static double[] Vector2ToArray(Vector2 value)
        {
            return new[] { (double)value.x, value.y };
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

        private static double[] IntrinsicsToArray(PassthroughCameraAccess cameraAccess)
        {
            PassthroughCameraAccess.CameraIntrinsics intrinsics = cameraAccess.Intrinsics;
            Vector2Int resolution = cameraAccess.CurrentResolution;
            Vector2Int sensorResolution = intrinsics.SensorResolution;

            double fx = intrinsics.FocalLength.x * resolution.x / sensorResolution.x;
            double fy = intrinsics.FocalLength.y * resolution.y / sensorResolution.y;
            double cx = intrinsics.PrincipalPoint.x * resolution.x / sensorResolution.x;
            double cy = intrinsics.PrincipalPoint.y * resolution.y / sensorResolution.y;

            return new[] { fx, fy, cx, cy };
        }

        private void OnDestroy()
        {
            if (isRecording)
            {
                StopRecording();
            }
            else
            {
                CloseJsonlWriters();
            }

            leftRecorder?.Dispose();
            rightRecorder?.Dispose();
        }

        private void OnDisable()
        {
            if (isRecording)
            {
                StopRecording();
            }
            else
            {
                CloseJsonlWriters();
            }
        }

        private void OnApplicationPause(bool pause)
        {
            if (pause && isRecording)
            {
                StopRecording();
            }
        }

        [Serializable]
        private sealed class RecordMetadata
        {
            public string startTimeUtc;
            public string schemaVersion;
            public string outputDirectory;
            public double durationSeconds;
            public int fps;
            public string poseConvention = "xyzqwqxqyqz";
            public string intrinsicsConvention = "fxfycxcy";
            public string trajectoryFrame;
            public string trajectorySampleSource;
            public string trajectoryContents;
            public string cameraPoseConvention;
            public string gazePointConvention;
            public string gazeProjectionConvention;
            public string cameraDistortionModel;
            public string realtimeTelemetryProtocol;
            public string realtimeTelemetryContents;
            public bool realtimeTelemetryEnabled;
            public string realtimeTelemetryHost;
            public int realtimeTelemetryPort;
            public string gazeSource;
            public string leftVideoFileName;
            public string rightVideoFileName;
            public string leftFrameMetadataFileName;
            public string rightFrameMetadataFileName;
            public string trajectoryFileName;
            public string leftVideoSha256;
            public string rightVideoSha256;
            public bool hasLeftCameraAccess;
            public bool hasRightCameraAccess;
            public bool recordRightCamera;
            public int leftFrameCount;
            public int rightFrameCount;
            public int trajectorySampleCount;
            public int gazeSampleCount;
            public int gazeHitSampleCount;
            public int gazeHitMissSampleCount;
            public double gazeSampleRatio;
            public double gazeHitSampleRatio;
            public double leftFrameTrajectoryCoverageRatio;
            public double rightFrameTrajectoryCoverageRatio;
            public double meanAbsLeftFrameTimeDeltaSeconds;
            public double meanAbsRightFrameTimeDeltaSeconds;
            public double maxAbsLeftFrameTimeDeltaSeconds;
            public double maxAbsRightFrameTimeDeltaSeconds;
            public string recordingQualitySummary;
            public double[] leftIntrinsics;
            public double[] rightIntrinsics;
            public int[] leftResolution;
            public int[] rightResolution;
            public bool recordSynchronizedTrajectory;
            public bool allowHeadCenterGazeFallback;
            public bool recordHmdEyePoses;
            public string trajectorySyncConvention;

            public void Reset()
            {
                startTimeUtc = null;
                schemaVersion = null;
                outputDirectory = null;
                durationSeconds = 0;
                fps = 0;
                leftVideoFileName = null;
                rightVideoFileName = null;
                leftFrameMetadataFileName = null;
                rightFrameMetadataFileName = null;
                trajectoryFileName = null;
                leftVideoSha256 = null;
                rightVideoSha256 = null;
                hasLeftCameraAccess = false;
                hasRightCameraAccess = false;
                recordRightCamera = false;
                leftFrameCount = 0;
                rightFrameCount = 0;
                trajectorySampleCount = 0;
                gazeSampleCount = 0;
                gazeHitSampleCount = 0;
                gazeHitMissSampleCount = 0;
                gazeSampleRatio = 0;
                gazeHitSampleRatio = 0;
                leftFrameTrajectoryCoverageRatio = 0;
                rightFrameTrajectoryCoverageRatio = 0;
                meanAbsLeftFrameTimeDeltaSeconds = 0;
                meanAbsRightFrameTimeDeltaSeconds = 0;
                maxAbsLeftFrameTimeDeltaSeconds = 0;
                maxAbsRightFrameTimeDeltaSeconds = 0;
                recordingQualitySummary = null;
                leftIntrinsics = null;
                rightIntrinsics = null;
                leftResolution = null;
                rightResolution = null;
                trajectoryFrame = null;
                trajectorySampleSource = null;
                trajectoryContents = null;
                cameraPoseConvention = null;
                gazePointConvention = null;
                gazeProjectionConvention = null;
                cameraDistortionModel = null;
                realtimeTelemetryProtocol = null;
                realtimeTelemetryContents = null;
                realtimeTelemetryEnabled = false;
                realtimeTelemetryHost = null;
                realtimeTelemetryPort = 0;
                gazeSource = null;
                trajectorySyncConvention = null;
                recordSynchronizedTrajectory = false;
                allowHeadCenterGazeFallback = false;
                recordHmdEyePoses = false;
            }

            public string ToJson()
            {
                return JsonUtility.ToJson(this, true);
            }
        }

        [Serializable]
        private sealed class FrameMetadata
        {
            public int frameIndex;
            public double cameraTimestampSeconds;
            public double unityTimestampSeconds;
            public double[] pose;
        }

        [Serializable]
        public sealed class TrajectorySample
        {
            public int sampleIndex;
            public int liveSampleIndex;
            public bool isRecording;
            public string telemetryMode;
            public double unityTimestampSeconds;
            public double recordingTimestampSeconds;
            public bool hasLeftCameraPose;
            public bool hasRightCameraPose;
            public int nearestLeftFrameIndex;
            public int nearestRightFrameIndex;
            public double nearestLeftFrameTimeDeltaSeconds;
            public double nearestRightFrameTimeDeltaSeconds;
            public double leftCameraTimestampSeconds;
            public double rightCameraTimestampSeconds;
            public double[] leftCameraPose;
            public double[] rightCameraPose;
            public double[] leftPassthroughCameraPose;
            public double[] rightPassthroughCameraPose;
            public bool hasLeftEyePose;
            public bool hasRightEyePose;
            public string leftEyePoseSource;
            public string rightEyePoseSource;
            public double[] leftEyePose;
            public double[] rightEyePose;
            public double[] leftEyePosition;
            public double[] rightEyePosition;
            public QuestRecordingTelemetrySender.ControllerTelemetry leftController;
            public QuestRecordingTelemetrySender.ControllerTelemetry rightController;
            public bool hasGaze;
            public string gazeSource;
            public string gazeMissReason;
            public double[] gazePitchYaw;
            public double[] gazePitchYawBias;
            public double[] gazeBiasedPitchYaw;
            public double[] gazeRayOrigin;
            public double[] gazeRayDirection;
            public double[] gazeFallbackPointWorld;
            public bool hasGazeHit;
            public string gazeHitMissReason;
            public double[] gazePointWorld;
            public double[] gazePoint3DWorld;
            public string gazePoint3DSource;
            public bool hasLeftGazePointProjection;
            public double[] leftGazePointViewport;
            public double[] leftGazePointPixel;
            public bool leftGazePointInImage;
            public double leftGazePointCameraZ;
            public string leftGazePointProjectionSource;
            public string leftGazePointProjectionMissReason;
            public bool hasRightGazePointProjection;
            public double[] rightGazePointViewport;
            public double[] rightGazePointPixel;
            public bool rightGazePointInImage;
            public double rightGazePointCameraZ;
            public string rightGazePointProjectionSource;
            public string rightGazePointProjectionMissReason;
        }

        private struct CameraProjection
        {
            public Vector2 viewport;
            public Vector2 pixel;
            public bool inImage;
            public double cameraZ;
        }
    }
}
