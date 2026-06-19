using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Oculus.Interaction.Input;
using UnityEngine;
using UnityEngine.XR;

namespace EyeTracking.Recording
{
    [DisallowMultipleComponent]
    public sealed class QuestRecordingTelemetrySender : MonoBehaviour
    {
        public const string ProtocolName = "quest_recording_telemetry_v1";

        [Header("UDP Telemetry")]
        [SerializeField] private bool sendTelemetry = true;
        [SerializeField] private string host = "10.128.0.227";
        [SerializeField] private int port = 9100;
        [SerializeField] private bool autoResolveTrackingSpace = true;
        [SerializeField] private Transform trackingSpace;
        [SerializeField] private bool autoResolveControllerAnchors = true;
        [SerializeField] private Transform leftControllerAnchor;
        [SerializeField] private Transform rightControllerAnchor;
        [SerializeField] private bool autoResolveInteractionControllerRefs = true;
        [SerializeField, Min(1)] private int maxPacketsPerSecond = 90;
        [SerializeField] private bool sendLifecycleMessages = true;
        [SerializeField] private bool logSendErrors = true;
        [SerializeField] private bool logTelemetryStatus = true;
        [SerializeField, Min(1)] private int sampleStatusLogInterval = 120;

        private UdpClient udpClient;
        private IPEndPoint remoteEndPoint;
        private float nextSampleAllowedTime;
        private int telemetrySequence;
        private int datagramsSent;
        private int sampleMessagesSent;
        private string currentRecordId;
        private string currentOutputDirectory;
        private bool isRecording;
        private bool warnedMissingHost;
        private bool warnedSendError;
        private bool loggedFirstSample;
        private string lastLeftControllerSource;
        private string lastRightControllerSource;
        private ControllerRef[] interactionControllerRefs;
        private float nextInteractionControllerRefRefreshTime;

        public bool SendTelemetry
        {
            get { return sendTelemetry; }
            set { sendTelemetry = value; }
        }

        public string Host
        {
            get { return host; }
            set
            {
                if (host == value)
                {
                    return;
                }

                host = value;
                ReopenSocket();
            }
        }

        public int Port
        {
            get { return port; }
            set
            {
                if (port == value)
                {
                    return;
                }

                port = value;
                ReopenSocket();
            }
        }

        public bool IsRecording => isRecording;

        public bool CanSendSampleNow()
        {
            if (!sendTelemetry)
            {
                return false;
            }

            return maxPacketsPerSecond <= 0 || Time.unscaledTime >= nextSampleAllowedTime;
        }

        public void NotifyRecordingStarted(string outputDirectory, string recordId, DateTime startTimeUtc)
        {
            currentOutputDirectory = outputDirectory;
            currentRecordId = recordId;
            isRecording = true;
            telemetrySequence = 0;
            datagramsSent = 0;
            sampleMessagesSent = 0;
            nextSampleAllowedTime = 0f;
            warnedSendError = false;
            loggedFirstSample = false;
            lastLeftControllerSource = null;
            lastRightControllerSource = null;

            ResolveTrackingSpace();
            ResolveControllerAnchors();
            if (logTelemetryStatus)
            {
                Debug.Log(
                    "[QuestRecordingTelemetrySender] Recording telemetry starting " +
                    $"recordId={currentRecordId} target={host}:{port} " +
                    $"trackingSpace={TransformName(trackingSpace)} " +
                    $"leftAnchor={TransformName(leftControllerAnchor)} rightAnchor={TransformName(rightControllerAnchor)}",
                    this);
            }

            if (!sendLifecycleMessages)
            {
                return;
            }

            LifecycleTelemetryMessage message = new LifecycleTelemetryMessage
            {
                protocol = ProtocolName,
                type = "recording_start",
                sequence = telemetrySequence++,
                isRecording = true,
                telemetryMode = "recording",
                recordId = currentRecordId,
                outputDirectory = currentOutputDirectory,
                unityTimestampSeconds = Time.realtimeSinceStartupAsDouble,
                recordingTimestampSeconds = 0.0,
                startTimeUtc = startTimeUtc.ToString("o")
            };
            SendJson(JsonUtility.ToJson(message, false));
        }

        public void NotifyRecordingStopped()
        {
            if (sendLifecycleMessages)
            {
                LifecycleTelemetryMessage message = new LifecycleTelemetryMessage
                {
                    protocol = ProtocolName,
                    type = "recording_stop",
                    sequence = telemetrySequence++,
                    isRecording = false,
                    telemetryMode = "live_preview",
                    recordId = currentRecordId,
                    outputDirectory = currentOutputDirectory,
                    unityTimestampSeconds = Time.realtimeSinceStartupAsDouble
                };
                SendJson(JsonUtility.ToJson(message, false));
            }

            currentRecordId = null;
            currentOutputDirectory = null;
            isRecording = false;

            if (logTelemetryStatus)
            {
                Debug.Log(
                    "[QuestRecordingTelemetrySender] Recording telemetry stopped. " +
                    $"datagramsSent={datagramsSent} sampleMessagesSent={sampleMessagesSent}",
                    this);
            }
        }

        public void SendSample(QuestCameraRecorder.TrajectorySample sample)
        {
            if (!sendTelemetry || sample == null)
            {
                return;
            }

            bool sampleIsRecording = sample.isRecording || isRecording;
            string telemetryMode = string.IsNullOrEmpty(sample.telemetryMode)
                ? (sampleIsRecording ? "recording" : "live_preview")
                : sample.telemetryMode;

            if (maxPacketsPerSecond > 0)
            {
                float now = Time.unscaledTime;
                if (now < nextSampleAllowedTime)
                {
                    return;
                }

                nextSampleAllowedTime = now + 1f / maxPacketsPerSecond;
            }

            ResolveTrackingSpace();
            ControllerTelemetry leftController = sample.leftController ?? CaptureControllerTelemetry(XRNode.LeftHand);
            ControllerTelemetry rightController = sample.rightController ?? CaptureControllerTelemetry(XRNode.RightHand);
            TelemetrySampleMessage message = new TelemetrySampleMessage
            {
                protocol = ProtocolName,
                type = "sample",
                sequence = telemetrySequence++,
                isRecording = sampleIsRecording,
                telemetryMode = telemetryMode,
                liveSampleIndex = sample.liveSampleIndex,
                recordId = sampleIsRecording ? currentRecordId : null,
                outputDirectory = sampleIsRecording ? currentOutputDirectory : null,
                sampleIndex = sample.sampleIndex,
                unityTimestampSeconds = sample.unityTimestampSeconds,
                recordingTimestampSeconds = sample.recordingTimestampSeconds,
                hasGaze = sample.hasGaze,
                gazeSource = sample.gazeSource,
                hasGazeHit = sample.hasGazeHit,
                gazePointWorld = sample.gazePointWorld,
                gazePoint3DWorld = sample.gazePoint3DWorld,
                gazePoint3DSource = sample.gazePoint3DSource,
                gazeRayOrigin = sample.gazeRayOrigin,
                gazeRayDirection = sample.gazeRayDirection,
                nearestLeftFrameIndex = sample.nearestLeftFrameIndex,
                nearestRightFrameIndex = sample.nearestRightFrameIndex,
                nearestLeftFrameTimeDeltaSeconds = sample.nearestLeftFrameTimeDeltaSeconds,
                nearestRightFrameTimeDeltaSeconds = sample.nearestRightFrameTimeDeltaSeconds,
                leftCameraPose = sample.leftCameraPose,
                rightCameraPose = sample.rightCameraPose,
                hasLeftEyePose = sample.hasLeftEyePose,
                hasRightEyePose = sample.hasRightEyePose,
                leftEyePoseSource = sample.leftEyePoseSource,
                rightEyePoseSource = sample.rightEyePoseSource,
                leftEyePose = sample.leftEyePose,
                rightEyePose = sample.rightEyePose,
                leftEyePosition = sample.leftEyePosition,
                rightEyePosition = sample.rightEyePosition,
                leftController = leftController,
                rightController = rightController
            };

            if (SendJson(JsonUtility.ToJson(message, false)))
            {
                sampleMessagesSent++;
                LogSampleStatusIfNeeded(message);
            }
        }

        public ControllerTelemetry CaptureControllerTelemetry(XRNode node)
        {
            ResolveTrackingSpace();
            return BuildControllerTelemetry(node);
        }

        private ControllerTelemetry BuildControllerTelemetry(XRNode node)
        {
            ControllerTelemetry telemetry = new ControllerTelemetry
            {
                handedness = node == XRNode.LeftHand ? "left" : "right"
            };

            if (TryGetOvrControllerPose(node, out Pose pose, out bool positionTracked, out bool rotationTracked))
            {
                FillTrackedPose(telemetry, "OVRInput", pose, positionTracked, rotationTracked);
                FillControllerInputTelemetry(telemetry, node);
                return telemetry;
            }

            if (TryGetXrControllerPose(node, out pose, out positionTracked, out rotationTracked))
            {
                FillTrackedPose(telemetry, "XRInputDevice", pose, positionTracked, rotationTracked);
                FillControllerInputTelemetry(telemetry, node);
                return telemetry;
            }

            if (TryGetInteractionControllerRefPose(node, out pose, out positionTracked, out rotationTracked))
            {
                FillTrackedPose(telemetry, "InteractionSDKControllerRef", pose, positionTracked, rotationTracked);
                FillControllerInputTelemetry(telemetry, node);
                return telemetry;
            }

            if (TryGetAnchorControllerPose(node, out pose, out positionTracked, out rotationTracked))
            {
                FillTrackedPose(telemetry, "OVRCameraRigAnchor", pose, positionTracked, rotationTracked);
                FillControllerInputTelemetry(telemetry, node);
                return telemetry;
            }

            FillControllerInputTelemetry(telemetry, node);
            telemetry.hasPose = false;
            telemetry.source = "missing";
            telemetry.missingReason = BuildControllerMissingReason(node);
            telemetry.positionTracked = false;
            telemetry.rotationTracked = false;
            telemetry.position = null;
            telemetry.rotation = null;
            telemetry.pose = null;
            return telemetry;
        }

        private static void FillTrackedPose(
            ControllerTelemetry telemetry,
            string source,
            Pose pose,
            bool positionTracked,
            bool rotationTracked)
        {
            telemetry.hasPose = true;
            telemetry.source = source;
            telemetry.missingReason = null;
            telemetry.positionTracked = positionTracked;
            telemetry.rotationTracked = rotationTracked;
            telemetry.position = Vector3ToArray(pose.position);
            telemetry.rotation = QuaternionToArray(pose.rotation);
            telemetry.pose = PoseToArray(pose);
        }

        private static void FillControllerInputTelemetry(ControllerTelemetry telemetry, XRNode node)
        {
            OVRInput.Controller controller = OvrControllerForNode(node);
            bool isRight = node == XRNode.RightHand;
            OVRInput.RawButton rawIndex = isRight ? OVRInput.RawButton.RIndexTrigger : OVRInput.RawButton.LIndexTrigger;
            OVRInput.RawButton rawHand = isRight ? OVRInput.RawButton.RHandTrigger : OVRInput.RawButton.LHandTrigger;
            OVRInput.RawAxis1D rawIndexAxis = isRight ? OVRInput.RawAxis1D.RIndexTrigger : OVRInput.RawAxis1D.LIndexTrigger;
            OVRInput.RawAxis1D rawHandAxis = isRight ? OVRInput.RawAxis1D.RHandTrigger : OVRInput.RawAxis1D.LHandTrigger;
            telemetry.indexTrigger = Mathf.Clamp01(OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, controller));
            telemetry.handTrigger = Mathf.Clamp01(OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, controller));
            telemetry.indexTrigger = Mathf.Clamp01(Mathf.Max(telemetry.indexTrigger, OVRInput.Get(rawIndexAxis, controller)));
            telemetry.handTrigger = Mathf.Clamp01(Mathf.Max(telemetry.handTrigger, OVRInput.Get(rawHandAxis, controller)));
            telemetry.indexTriggerPressed =
                OVRInput.Get(rawIndex, controller) ||
                OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, controller);
            telemetry.handTriggerPressed =
                OVRInput.Get(rawHand, controller) ||
                OVRInput.Get(OVRInput.Button.PrimaryHandTrigger, controller);
            telemetry.aButton =
                isRight &&
                (OVRInput.Get(OVRInput.RawButton.A, controller) || OVRInput.Get(OVRInput.Button.One, controller));
            telemetry.bButton =
                isRight &&
                (OVRInput.Get(OVRInput.RawButton.B, controller) || OVRInput.Get(OVRInput.Button.Two, controller));

            InputDevice device = InputDevices.GetDeviceAtXRNode(node);
            if (device.isValid)
            {
                if (device.TryGetFeatureValue(CommonUsages.trigger, out float xrTrigger))
                {
                    telemetry.indexTrigger = Mathf.Clamp01(Mathf.Max(telemetry.indexTrigger, xrTrigger));
                }
                if (device.TryGetFeatureValue(CommonUsages.grip, out float xrGrip))
                {
                    telemetry.handTrigger = Mathf.Clamp01(Mathf.Max(telemetry.handTrigger, xrGrip));
                }
                if (device.TryGetFeatureValue(CommonUsages.triggerButton, out bool xrTriggerButton))
                {
                    telemetry.indexTriggerPressed |= xrTriggerButton;
                }
                if (device.TryGetFeatureValue(CommonUsages.gripButton, out bool xrGripButton))
                {
                    telemetry.handTriggerPressed |= xrGripButton;
                }
                if (node == XRNode.RightHand && device.TryGetFeatureValue(CommonUsages.primaryButton, out bool xrPrimaryButton))
                {
                    telemetry.aButton |= xrPrimaryButton;
                }
                if (node == XRNode.RightHand && device.TryGetFeatureValue(CommonUsages.secondaryButton, out bool xrSecondaryButton))
                {
                    telemetry.bButton |= xrSecondaryButton;
                }
            }

        }

        private void LogSampleStatusIfNeeded(TelemetrySampleMessage message)
        {
            if (!logTelemetryStatus || message == null)
            {
                return;
            }

            string leftSource = ControllerSource(message.leftController);
            string rightSource = ControllerSource(message.rightController);
            bool controllerSourceChanged =
                lastLeftControllerSource != leftSource ||
                lastRightControllerSource != rightSource;
            bool shouldLog =
                !loggedFirstSample ||
                controllerSourceChanged ||
                sampleMessagesSent % sampleStatusLogInterval == 0;

            lastLeftControllerSource = leftSource;
            lastRightControllerSource = rightSource;
            if (!shouldLog)
            {
                return;
            }

            loggedFirstSample = true;
            Debug.Log(
                "[QuestRecordingTelemetrySender] Sent sample telemetry " +
                $"sampleMessagesSent={sampleMessagesSent} seq={message.sequence} " +
                $"mode={message.telemetryMode} recording={message.isRecording} " +
                $"gaze3D={message.gazePoint3DWorld != null} gazeHit={message.hasGazeHit} " +
                $"left={ControllerSummary(message.leftController)} right={ControllerSummary(message.rightController)}",
                this);
        }

        private bool TryGetOvrControllerPose(
            XRNode node,
            out Pose pose,
            out bool positionTracked,
            out bool rotationTracked)
        {
            OVRInput.Controller controller = OvrControllerForNode(node);

            if ((OVRInput.GetConnectedControllers() & controller) == 0)
            {
                pose = default;
                positionTracked = false;
                rotationTracked = false;
                return false;
            }

            positionTracked = OVRInput.GetControllerPositionTracked(controller);
            rotationTracked = OVRInput.GetControllerOrientationTracked(controller);
            if (!positionTracked && !rotationTracked)
            {
                pose = default;
                return false;
            }

            Vector3 localPosition = positionTracked
                ? OVRInput.GetLocalControllerPosition(controller)
                : Vector3.zero;
            Quaternion localRotation = rotationTracked
                ? OVRInput.GetLocalControllerRotation(controller)
                : Quaternion.identity;
            pose = ToWorldPose(localPosition, localRotation);
            return true;
        }

        private bool TryGetXrControllerPose(
            XRNode node,
            out Pose pose,
            out bool positionTracked,
            out bool rotationTracked)
        {
            InputDevice device = InputDevices.GetDeviceAtXRNode(node);
            if (!device.isValid)
            {
                pose = default;
                positionTracked = false;
                rotationTracked = false;
                return false;
            }

            positionTracked = device.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 position);
            rotationTracked = device.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rotation);
            if (!positionTracked && !rotationTracked)
            {
                pose = default;
                return false;
            }

            if (!positionTracked)
            {
                position = Vector3.zero;
            }

            if (!rotationTracked)
            {
                rotation = Quaternion.identity;
            }

            pose = ToWorldPose(position, rotation);
            return true;
        }

        private bool TryGetInteractionControllerRefPose(
            XRNode node,
            out Pose pose,
            out bool positionTracked,
            out bool rotationTracked)
        {
            pose = default;
            positionTracked = false;
            rotationTracked = false;

            RefreshInteractionControllerRefs();
            if (interactionControllerRefs == null || interactionControllerRefs.Length == 0)
            {
                return false;
            }

            Handedness handedness = node == XRNode.LeftHand ? Handedness.Left : Handedness.Right;
            for (int i = 0; i < interactionControllerRefs.Length; i++)
            {
                ControllerRef controller = interactionControllerRefs[i];
                if (controller == null || !controller.isActiveAndEnabled)
                {
                    continue;
                }

                try
                {
                    if (controller.Handedness != handedness ||
                        !controller.IsConnected ||
                        !controller.IsPoseValid ||
                        !controller.TryGetPose(out Pose controllerPose) ||
                        !IsFinitePose(controllerPose))
                    {
                        continue;
                    }

                    pose = controllerPose;
                    positionTracked = true;
                    rotationTracked = true;
                    return true;
                }
                catch (Exception)
                {
                    // Some ControllerRef instances may be present before their
                    // underlying Interaction SDK source is injected.
                }
            }

            return false;
        }

        private bool TryGetAnchorControllerPose(
            XRNode node,
            out Pose pose,
            out bool positionTracked,
            out bool rotationTracked)
        {
            ResolveControllerAnchors();
            Transform anchor = node == XRNode.LeftHand ? leftControllerAnchor : rightControllerAnchor;
            if (anchor == null || !anchor.gameObject.activeInHierarchy)
            {
                pose = default;
                positionTracked = false;
                rotationTracked = false;
                return false;
            }

            OVRInput.Controller controller = OvrControllerForNode(node);
            positionTracked = OVRInput.GetControllerPositionTracked(controller) ||
                OVRInput.GetControllerPositionValid(controller);
            rotationTracked = OVRInput.GetControllerOrientationTracked(controller) ||
                OVRInput.GetControllerOrientationValid(controller);
            if (!positionTracked && !rotationTracked)
            {
                pose = default;
                return false;
            }

            pose = new Pose(anchor.position, anchor.rotation);
            if (!IsFinitePose(pose))
            {
                pose = default;
                positionTracked = false;
                rotationTracked = false;
                return false;
            }

            return true;
        }

        private string BuildControllerMissingReason(XRNode node)
        {
            OVRInput.Controller controller = OvrControllerForNode(node);
            bool ovrConnected = (OVRInput.GetConnectedControllers() & controller) != 0;
            bool ovrPosition = ovrConnected &&
                (OVRInput.GetControllerPositionTracked(controller) || OVRInput.GetControllerPositionValid(controller));
            bool ovrRotation = ovrConnected &&
                (OVRInput.GetControllerOrientationTracked(controller) || OVRInput.GetControllerOrientationValid(controller));

            InputDevice device = InputDevices.GetDeviceAtXRNode(node);
            bool xrPosition = device.isValid && device.TryGetFeatureValue(CommonUsages.devicePosition, out _);
            bool xrRotation = device.isValid && device.TryGetFeatureValue(CommonUsages.deviceRotation, out _);

            int matchingRefs = 0;
            int connectedRefs = 0;
            int poseValidRefs = 0;
            Handedness handedness = node == XRNode.LeftHand ? Handedness.Left : Handedness.Right;
            if (interactionControllerRefs != null)
            {
                for (int i = 0; i < interactionControllerRefs.Length; i++)
                {
                    ControllerRef controllerRef = interactionControllerRefs[i];
                    if (controllerRef == null || !controllerRef.isActiveAndEnabled)
                    {
                        continue;
                    }

                    try
                    {
                        if (controllerRef.Handedness != handedness)
                        {
                            continue;
                        }

                        matchingRefs++;
                        if (controllerRef.IsConnected)
                        {
                            connectedRefs++;
                        }

                        if (controllerRef.IsPoseValid)
                        {
                            poseValidRefs++;
                        }
                    }
                    catch (Exception)
                    {
                        matchingRefs++;
                    }
                }
            }

            Transform anchor = node == XRNode.LeftHand ? leftControllerAnchor : rightControllerAnchor;
            bool anchorActive = anchor != null && anchor.gameObject.activeInHierarchy;

            return
                $"ovrConnected={ovrConnected};ovrPos={ovrPosition};ovrRot={ovrRotation};" +
                $"xrValid={device.isValid};xrPos={xrPosition};xrRot={xrRotation};" +
                $"interactionRefs={matchingRefs};interactionConnected={connectedRefs};interactionPoseValid={poseValidRefs};" +
                $"anchorActive={anchorActive}";
        }

        private Pose ToWorldPose(Vector3 localPosition, Quaternion localRotation)
        {
            ResolveTrackingSpace();
            return trackingSpace == null
                ? new Pose(localPosition, localRotation)
                : new Pose(trackingSpace.TransformPoint(localPosition), trackingSpace.rotation * localRotation);
        }

        private void ResolveTrackingSpace()
        {
            if (!autoResolveTrackingSpace || trackingSpace != null)
            {
                return;
            }

            OVRCameraRig rig = FindFirstObjectByType<OVRCameraRig>();
            if (rig != null && rig.trackingSpace != null)
            {
                trackingSpace = rig.trackingSpace;
                return;
            }

            GameObject xrOrigin = GameObject.Find("XR Origin");
            if (xrOrigin != null)
            {
                trackingSpace = xrOrigin.transform;
            }
        }

        private void ResolveControllerAnchors()
        {
            if (!autoResolveControllerAnchors ||
                leftControllerAnchor != null && rightControllerAnchor != null)
            {
                return;
            }

            OVRCameraRig rig = FindFirstObjectByType<OVRCameraRig>();
            if (rig != null)
            {
                if (trackingSpace == null && rig.trackingSpace != null)
                {
                    trackingSpace = rig.trackingSpace;
                }

                if (leftControllerAnchor == null)
                {
                    leftControllerAnchor = rig.leftControllerAnchor != null
                        ? rig.leftControllerAnchor
                        : FindChildRecursive(rig.transform, "LeftControllerAnchor");
                }

                if (rightControllerAnchor == null)
                {
                    rightControllerAnchor = rig.rightControllerAnchor != null
                        ? rig.rightControllerAnchor
                        : FindChildRecursive(rig.transform, "RightControllerAnchor");
                }
            }

            if (leftControllerAnchor == null)
            {
                GameObject left = GameObject.Find("LeftControllerAnchor");
                leftControllerAnchor = left != null ? left.transform : null;
            }

            if (rightControllerAnchor == null)
            {
                GameObject right = GameObject.Find("RightControllerAnchor");
                rightControllerAnchor = right != null ? right.transform : null;
            }
        }

        private void RefreshInteractionControllerRefs()
        {
            if (!autoResolveInteractionControllerRefs)
            {
                return;
            }

            if (interactionControllerRefs != null &&
                interactionControllerRefs.Length > 0 &&
                Time.unscaledTime < nextInteractionControllerRefRefreshTime)
            {
                return;
            }

            interactionControllerRefs = FindObjectsByType<ControllerRef>(FindObjectsSortMode.None);
            nextInteractionControllerRefRefreshTime = Time.unscaledTime + 1f;
        }

        private bool SendJson(string json)
        {
            if (!sendTelemetry)
            {
                return false;
            }

            if (!EnsureSocket())
            {
                return false;
            }

            try
            {
                byte[] bytes = Encoding.UTF8.GetBytes(json + "\n");
                udpClient.Send(bytes, bytes.Length);
                datagramsSent++;
                return true;
            }
            catch (Exception exception)
            {
                if (logSendErrors && !warnedSendError)
                {
                    warnedSendError = true;
                    Debug.LogWarning("[QuestRecordingTelemetrySender] UDP send failed: " + exception.Message, this);
                }

                ReopenSocket();
                return false;
            }
        }

        private bool EnsureSocket()
        {
            if (string.IsNullOrWhiteSpace(host))
            {
                if (!warnedMissingHost)
                {
                    warnedMissingHost = true;
                    Debug.LogWarning("[QuestRecordingTelemetrySender] Telemetry host is empty; UDP telemetry disabled.", this);
                }

                return false;
            }

            if (udpClient != null && remoteEndPoint != null)
            {
                return true;
            }

            try
            {
                IPAddress address;
                if (!IPAddress.TryParse(host.Trim(), out address))
                {
                    IPAddress[] addresses = Dns.GetHostAddresses(host.Trim());
                    address = addresses != null && addresses.Length > 0 ? addresses[0] : null;
                }

                if (address == null)
                {
                    throw new SocketException((int)SocketError.HostNotFound);
                }

                remoteEndPoint = new IPEndPoint(address, port);
                udpClient = new UdpClient();
                udpClient.Connect(remoteEndPoint);
                if (logTelemetryStatus)
                {
                    Debug.Log("[QuestRecordingTelemetrySender] UDP telemetry target opened: " + remoteEndPoint, this);
                }

                return true;
            }
            catch (Exception exception)
            {
                if (logSendErrors && !warnedSendError)
                {
                    warnedSendError = true;
                    Debug.LogWarning("[QuestRecordingTelemetrySender] Could not open UDP telemetry target " + host + ":" + port + ": " + exception.Message, this);
                }

                CloseSocket();
                return false;
            }
        }

        private void ReopenSocket()
        {
            CloseSocket();
            warnedMissingHost = false;
            warnedSendError = false;
        }

        private void CloseSocket()
        {
            if (udpClient == null)
            {
                return;
            }

            udpClient.Close();
            udpClient = null;
            remoteEndPoint = null;
        }

        private void OnDisable()
        {
            CloseSocket();
        }

        private void OnDestroy()
        {
            CloseSocket();
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

        private static double[] QuaternionToArray(Quaternion value)
        {
            return new[] { (double)value.w, value.x, value.y, value.z };
        }

        private static Transform FindChildRecursive(Transform root, string childName)
        {
            if (root == null || string.IsNullOrEmpty(childName))
            {
                return null;
            }

            if (root.name == childName)
            {
                return root;
            }

            for (int i = 0; i < root.childCount; i++)
            {
                Transform found = FindChildRecursive(root.GetChild(i), childName);
                if (found != null)
                {
                    return found;
                }
            }

            return null;
        }

        private static bool IsFinitePose(Pose pose)
        {
            return IsFiniteVector3(pose.position) &&
                IsFiniteFloat(pose.rotation.x) &&
                IsFiniteFloat(pose.rotation.y) &&
                IsFiniteFloat(pose.rotation.z) &&
                IsFiniteFloat(pose.rotation.w);
        }

        private static bool IsFiniteVector3(Vector3 value)
        {
            return IsFiniteFloat(value.x) &&
                IsFiniteFloat(value.y) &&
                IsFiniteFloat(value.z);
        }

        private static bool IsFiniteFloat(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static OVRInput.Controller OvrControllerForNode(XRNode node)
        {
            return node == XRNode.LeftHand
                ? OVRInput.Controller.LTouch
                : OVRInput.Controller.RTouch;
        }

        private static string TransformName(Transform transform)
        {
            return transform != null ? transform.name : "null";
        }

        private static string ControllerSource(ControllerTelemetry telemetry)
        {
            return telemetry != null && !string.IsNullOrEmpty(telemetry.source)
                ? telemetry.source
                : "null";
        }

        private static string ControllerSummary(ControllerTelemetry telemetry)
        {
            if (telemetry == null)
            {
                return "null";
            }

            string missing = string.IsNullOrEmpty(telemetry.missingReason)
                ? string.Empty
                : "/reason=" + telemetry.missingReason;
            return $"{telemetry.source}/hasPose={telemetry.hasPose}/pos={telemetry.positionTracked}/rot={telemetry.rotationTracked}{missing}";
        }

        [Serializable]
        private sealed class LifecycleTelemetryMessage
        {
            public string protocol;
            public string type;
            public int sequence;
            public bool isRecording;
            public string telemetryMode;
            public string recordId;
            public string outputDirectory;
            public double unityTimestampSeconds;
            public double recordingTimestampSeconds;
            public string startTimeUtc;
        }

        [Serializable]
        private sealed class TelemetrySampleMessage
        {
            public string protocol;
            public string type;
            public int sequence;
            public bool isRecording;
            public string telemetryMode;
            public int liveSampleIndex;
            public string recordId;
            public string outputDirectory;
            public int sampleIndex;
            public double unityTimestampSeconds;
            public double recordingTimestampSeconds;
            public bool hasGaze;
            public string gazeSource;
            public bool hasGazeHit;
            public double[] gazePointWorld;
            public double[] gazePoint3DWorld;
            public string gazePoint3DSource;
            public double[] gazeRayOrigin;
            public double[] gazeRayDirection;
            public int nearestLeftFrameIndex;
            public int nearestRightFrameIndex;
            public double nearestLeftFrameTimeDeltaSeconds;
            public double nearestRightFrameTimeDeltaSeconds;
            public double[] leftCameraPose;
            public double[] rightCameraPose;
            public bool hasLeftEyePose;
            public bool hasRightEyePose;
            public string leftEyePoseSource;
            public string rightEyePoseSource;
            public double[] leftEyePose;
            public double[] rightEyePose;
            public double[] leftEyePosition;
            public double[] rightEyePosition;
            public ControllerTelemetry leftController;
            public ControllerTelemetry rightController;
        }

        [Serializable]
        public sealed class ControllerTelemetry
        {
            public string handedness;
            public bool hasPose;
            public string source;
            public bool positionTracked;
            public bool rotationTracked;
            public string missingReason;
            public double[] position;
            public double[] rotation;
            public double[] pose;
            public float indexTrigger;
            public float handTrigger;
            public bool indexTriggerPressed;
            public bool handTriggerPressed;
            public bool aButton;
            public bool bButton;
        }
    }
}
