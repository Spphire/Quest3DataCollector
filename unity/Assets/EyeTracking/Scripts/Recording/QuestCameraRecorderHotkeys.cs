using UnityEngine;

namespace EyeTracking.Recording
{
    [DisallowMultipleComponent]
    public sealed class QuestCameraRecorderHotkeys : MonoBehaviour
    {
        [SerializeField] private QuestCameraRecorder recorder;
        [SerializeField] private QuestPcCalibrationRecorder calibrationRecorder;
        [SerializeField] private GameObject recordDot;
        [SerializeField] private GameObject calibrationDot;
        [SerializeField] private string recordDotName = "Record_Dot";
        [SerializeField] private string calibrationDotName = "Calibration_Record_Dot";
        [SerializeField] private bool findRecordDotByName = true;
        [SerializeField] private bool createRecordDotIfMissing = true;
        [SerializeField] private bool ignoreInputWhileSystemMenuHeld = true;
        [SerializeField] private bool logControllerInputStatus = true;
        [SerializeField, Min(0.5f)] private float controllerInputStatusIntervalSeconds = 3f;
        [SerializeField] private Vector3 recordDotLocalPosition = new Vector3(0.22f, 0.16f, 0.7f);
        [SerializeField] private Vector3 calibrationDotLocalPosition = new Vector3(0.18f, 0.16f, 0.7f);
        [SerializeField] private float recordDotScale = 0.025f;

        private bool lastDotVisible;
        private bool lastCalibrationDotVisible;
        private float nextControllerInputStatusTime;

        private void Awake()
        {
            ResolveReferences();
            SetRecordDotVisible(false);
            SetCalibrationDotVisible(false);
        }

        private void Update()
        {
            ResolveReferences();
            LogControllerInputStatus();

            if (ignoreInputWhileSystemMenuHeld && QuestRecordingInputGuard.ShouldIgnoreRecordingInput())
            {
                SetRecordDotVisible(recorder != null && recorder.IsRecording);
                SetCalibrationDotVisible(calibrationRecorder != null && calibrationRecorder.IsRecording);
                return;
            }

            if (RightAButtonDown())
            {
                recorder?.ToggleRecording();
            }

            if (RightBButtonDown())
            {
                calibrationRecorder?.ToggleCalibrationRecording();
            }

            SetRecordDotVisible(recorder != null && recorder.IsRecording);
            SetCalibrationDotVisible(calibrationRecorder != null && calibrationRecorder.IsRecording);
        }

        private void ResolveReferences()
        {
            if (recorder == null)
            {
                recorder = FindFirstObjectByType<QuestCameraRecorder>();
            }

            if (calibrationRecorder == null)
            {
                calibrationRecorder = FindFirstObjectByType<QuestPcCalibrationRecorder>();
            }

            if (recordDot == null && findRecordDotByName && !string.IsNullOrEmpty(recordDotName))
            {
                recordDot = FindSceneObjectByName(recordDotName);
            }

            if (calibrationDot == null && findRecordDotByName && !string.IsNullOrEmpty(calibrationDotName))
            {
                calibrationDot = FindSceneObjectByName(calibrationDotName);
            }

            if (recordDot == null && createRecordDotIfMissing)
            {
                recordDot = CreateRecordDot(recordDotName, recordDotLocalPosition, Color.red);
            }

            if (calibrationDot == null && createRecordDotIfMissing)
            {
                calibrationDot = CreateRecordDot(calibrationDotName, calibrationDotLocalPosition, new Color(0.15f, 0.55f, 1f, 1f));
            }
        }

        private void SetRecordDotVisible(bool visible)
        {
            if (recordDot == null || (recordDot.activeSelf == visible && lastDotVisible == visible))
            {
                lastDotVisible = visible;
                return;
            }

            recordDot.SetActive(visible);
            lastDotVisible = visible;
        }

        private void SetCalibrationDotVisible(bool visible)
        {
            if (calibrationDot == null || (calibrationDot.activeSelf == visible && lastCalibrationDotVisible == visible))
            {
                lastCalibrationDotVisible = visible;
                return;
            }

            calibrationDot.SetActive(visible);
            lastCalibrationDotVisible = visible;
        }

        private static bool RightAButtonDown()
        {
            return OVRInput.GetDown(OVRInput.RawButton.A, OVRInput.Controller.RTouch) ||
                   OVRInput.GetDown(OVRInput.RawButton.A, OVRInput.Controller.All) ||
                   OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch) ||
                   OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.All);
        }

        private static bool RightBButtonDown()
        {
            return OVRInput.GetDown(OVRInput.RawButton.B, OVRInput.Controller.RTouch) ||
                   OVRInput.GetDown(OVRInput.RawButton.B, OVRInput.Controller.All) ||
                   OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.RTouch) ||
                   OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.All);
        }

        private void LogControllerInputStatus()
        {
            if (!logControllerInputStatus || Time.unscaledTime < nextControllerInputStatusTime)
            {
                return;
            }

            nextControllerInputStatusTime = Time.unscaledTime + controllerInputStatusIntervalSeconds;
            OVRInput.Controller connected = OVRInput.GetConnectedControllers();
            bool rTouchConnected = (connected & OVRInput.Controller.RTouch) != 0;
            bool anyTouchConnected =
                (connected & (OVRInput.Controller.LTouch | OVRInput.Controller.RTouch | OVRInput.Controller.Touch)) != 0;
            bool aHeld = OVRInput.Get(OVRInput.RawButton.A, OVRInput.Controller.All) ||
                         OVRInput.Get(OVRInput.Button.One, OVRInput.Controller.All);
            bool bHeld = OVRInput.Get(OVRInput.RawButton.B, OVRInput.Controller.All) ||
                         OVRInput.Get(OVRInput.Button.Two, OVRInput.Controller.All);
            bool rPos = OVRInput.GetControllerPositionTracked(OVRInput.Controller.RTouch);
            bool rRot = OVRInput.GetControllerOrientationTracked(OVRInput.Controller.RTouch);
            Debug.Log(
                "[QuestCameraRecorderHotkeys] input " +
                $"connected={connected} rTouchConnected={rTouchConnected} anyTouchConnected={anyTouchConnected} " +
                $"rPos={rPos} rRot={rRot} aHeld={aHeld} bHeld={bHeld} " +
                $"recording={recorder != null && recorder.IsRecording} calibration={calibrationRecorder != null && calibrationRecorder.IsRecording}",
                this);
        }

        private static GameObject FindSceneObjectByName(string objectName)
        {
            GameObject activeObject = GameObject.Find(objectName);
            if (activeObject != null)
            {
                return activeObject;
            }

            GameObject[] allObjects = Resources.FindObjectsOfTypeAll<GameObject>();
            for (int i = 0; i < allObjects.Length; i++)
            {
                GameObject candidate = allObjects[i];
                if (candidate != null &&
                    candidate.name == objectName &&
                    candidate.scene.IsValid())
                {
                    return candidate;
                }
            }

            return null;
        }

        private GameObject CreateRecordDot(string objectName, Vector3 localPosition, Color color)
        {
            GameObject dot = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            dot.name = string.IsNullOrEmpty(objectName) ? "Record_Dot" : objectName;

            Transform parent = Camera.main != null ? Camera.main.transform : transform;
            dot.transform.SetParent(parent, false);
            dot.transform.localPosition = localPosition;
            dot.transform.localRotation = Quaternion.identity;
            dot.transform.localScale = Vector3.one * recordDotScale;

            Renderer renderer = dot.GetComponent<Renderer>();
            if (renderer != null)
            {
                Material material = new Material(Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Unlit/Color"))
                {
                    color = color
                };
                if (material.HasProperty("_BaseColor"))
                {
                    material.SetColor("_BaseColor", color);
                }

                renderer.material = material;
            }

            dot.SetActive(false);
            return dot;
        }
    }
}
