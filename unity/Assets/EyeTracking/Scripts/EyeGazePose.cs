using Anaglyph.XRTemplate;
using UnityEngine;
using UnityEngine.XR;

public class EyeGazePose : MonoBehaviour
{
    private const string BiasPitchKey = "EyeGazePose.OscPitchBiasDegrees";
    private const string BiasYawKey = "EyeGazePose.OscYawBiasDegrees";

    private EnvironmentMapper.RayResult hit;

    public float checkDistance = 10f;
    public GameObject vizObj;

    [Header("OSC")]
    [SerializeField] private EyeTracking.PaperTrackerOscReceiver oscReceiver;
    [SerializeField] private bool useOscWhenConnected = true;
    [SerializeField] private bool findOscReceiverOnStart = true;
    [SerializeField] private float rayOffset = 0.1f;

    [Header("OSC Bias Fine Tune")]
    [SerializeField] private bool enableControllerBiasFineTune = true;
    [SerializeField, Min(0f)] private float biasDegreesPerSecond = 8f;
    [SerializeField, Range(0f, 1f)] private float rightStickDeadzone = 0.15f;
    [SerializeField] private bool persistOscBias = true;
    [SerializeField] private Vector2 oscPitchYawBiasDegrees;
    [SerializeField] private Vector2 latestBiasedPitchYaw;

    [Header("Fallback")]
    public Vector2 eyeGazePoint2f;

    private Camera mainCam;
    private Renderer[] selfVizRenderers;
    private bool hasVizPoint;
    private bool loadedPersistentBias;
    private bool wasLeftXPressed;
    private bool wasLeftYPressed;
    private bool biasDirty;

    public Vector2 OscPitchYawBiasDegrees => oscPitchYawBiasDegrees;
    public Vector2 LatestBiasedPitchYaw => latestBiasedPitchYaw;
    public bool HasVizPoint => vizObj != null && hasVizPoint;
    public Vector3 VizPointWorld => vizObj != null ? vizObj.transform.position : Vector3.zero;

    private void Start()
    {
        if (oscReceiver == null && findOscReceiverOnStart)
        {
            oscReceiver = FindFirstObjectByType<EyeTracking.PaperTrackerOscReceiver>();
        }

        LoadPersistentBias();

        if (vizObj == gameObject)
        {
            selfVizRenderers = vizObj.GetComponentsInChildren<Renderer>(true);
        }
    }

    private void Update()
    {
        UpdateOscBiasFineTune();

        if (!TryGetRay(out Ray ray))
        {
            hasVizPoint = false;
            SetVizActive(false);
            return;
        }

        if (EnvironmentMapper.Raycast(ray, checkDistance, out hit, EnvironmentMapper.RaycastMode.Negative))
        {
            hasVizPoint = true;
            SetVizActive(true);

            if (vizObj != null)
            {
                vizObj.transform.position = hit.point;
            }
        }
        else
        {
            hasVizPoint = false;
            SetVizActive(false);
        }
    }

    private bool TryGetRay(out Ray ray)
    {
        if (useOscWhenConnected && oscReceiver != null && oscReceiver.IsConnected)
        {
            Transform origin = GetRayOrigin();
            ray = new Ray(origin.position, origin.TransformDirection(GetBiasedOscDirectionLocal()));
            ray.origin += ray.direction * rayOffset;
            return true;
        }

        return TryGetFallbackScreenPointRay(out ray);
    }

    public bool TryGetBiasedOscRay(out Ray ray)
    {
        if (!useOscWhenConnected || oscReceiver == null || !oscReceiver.IsConnected)
        {
            ray = default(Ray);
            return false;
        }

        Transform origin = GetRayOrigin();
        ray = new Ray(origin.position, origin.TransformDirection(GetBiasedOscDirectionLocal()));
        ray.origin += ray.direction * rayOffset;
        return true;
    }

    public bool TryGetFallbackScreenPointRay(out Ray ray)
    {
        mainCam = mainCam != null ? mainCam : Camera.main;
        if (mainCam == null)
        {
            ray = default(Ray);
            return false;
        }

        Vector3 screenPos = new Vector3(
            eyeGazePoint2f.x * Screen.width,
            eyeGazePoint2f.y * Screen.height,
            0f
        );

        ray = mainCam.ScreenPointToRay(screenPos);
        ray.origin += ray.direction * rayOffset;
        return true;
    }

    public Vector2 GetBiasedOscPitchYaw()
    {
        Vector2 rawPitchYaw = oscReceiver != null ? oscReceiver.CenterPitchYaw : Vector2.zero;
        latestBiasedPitchYaw = rawPitchYaw + oscPitchYawBiasDegrees;
        return latestBiasedPitchYaw;
    }

    private Vector3 GetBiasedOscDirectionLocal()
    {
        Vector2 pitchYaw = GetBiasedOscPitchYaw();
        return (Quaternion.Euler(pitchYaw.x, pitchYaw.y, 0f) * Vector3.forward).normalized;
    }

    private void UpdateOscBiasFineTune()
    {
        if (!enableControllerBiasFineTune)
        {
            return;
        }

        bool leftXPressed = LeftXPressed();
        bool leftYPressed = LeftYPressed();

        if (leftXPressed)
        {
            Vector2 stick = RightThumbstick();
            if (stick.sqrMagnitude >= rightStickDeadzone * rightStickDeadzone)
            {
                oscPitchYawBiasDegrees += new Vector2(-stick.y, stick.x) * biasDegreesPerSecond * Time.deltaTime;
                latestBiasedPitchYaw = oscReceiver != null ? oscReceiver.CenterPitchYaw + oscPitchYawBiasDegrees : oscPitchYawBiasDegrees;
                biasDirty = true;
            }
        }
        else if (wasLeftXPressed && biasDirty)
        {
            SavePersistentBias();
            biasDirty = false;
        }

        if (leftYPressed && !wasLeftYPressed)
        {
            if (oscPitchYawBiasDegrees != Vector2.zero)
            {
                oscPitchYawBiasDegrees = Vector2.zero;
                latestBiasedPitchYaw = oscReceiver != null ? oscReceiver.CenterPitchYaw : Vector2.zero;
            }

            SavePersistentBias();
            biasDirty = false;
        }

        wasLeftXPressed = leftXPressed;
        wasLeftYPressed = leftYPressed;
    }

    private bool LeftXPressed()
    {
        if (OVRInput.Get(OVRInput.RawButton.X, OVRInput.Controller.LTouch) ||
            OVRInput.Get(OVRInput.Button.Three, OVRInput.Controller.LTouch))
        {
            return true;
        }

        InputDevice device = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        return device.isValid &&
               device.TryGetFeatureValue(CommonUsages.primaryButton, out bool pressed) &&
               pressed;
    }

    private bool LeftYPressed()
    {
        if (OVRInput.Get(OVRInput.RawButton.Y, OVRInput.Controller.LTouch) ||
            OVRInput.Get(OVRInput.Button.Four, OVRInput.Controller.LTouch))
        {
            return true;
        }

        InputDevice device = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        return device.isValid &&
               device.TryGetFeatureValue(CommonUsages.secondaryButton, out bool pressed) &&
               pressed;
    }

    private Vector2 RightThumbstick()
    {
        Vector2 stick = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick, OVRInput.Controller.RTouch);
        if (stick.sqrMagnitude > 0f)
        {
            return stick;
        }

        stick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.RTouch);
        if (stick.sqrMagnitude > 0f)
        {
            return stick;
        }

        InputDevice device = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        if (device.isValid &&
            device.TryGetFeatureValue(CommonUsages.primary2DAxis, out stick))
        {
            return stick;
        }

        return Vector2.zero;
    }

    private void LoadPersistentBias()
    {
        if (loadedPersistentBias || !persistOscBias)
        {
            return;
        }

        loadedPersistentBias = true;
        oscPitchYawBiasDegrees = new Vector2(
            PlayerPrefs.GetFloat(BiasPitchKey, oscPitchYawBiasDegrees.x),
            PlayerPrefs.GetFloat(BiasYawKey, oscPitchYawBiasDegrees.y));
        latestBiasedPitchYaw = oscReceiver != null ? oscReceiver.CenterPitchYaw + oscPitchYawBiasDegrees : oscPitchYawBiasDegrees;
    }

    private void SavePersistentBias()
    {
        if (!persistOscBias)
        {
            return;
        }

        PlayerPrefs.SetFloat(BiasPitchKey, oscPitchYawBiasDegrees.x);
        PlayerPrefs.SetFloat(BiasYawKey, oscPitchYawBiasDegrees.y);
        PlayerPrefs.Save();
    }

    private void OnApplicationPause(bool pause)
    {
        if (pause)
        {
            SavePersistentBias();
        }
    }

    private void OnDisable()
    {
        SavePersistentBias();
    }

    private Transform GetRayOrigin()
    {
        mainCam = mainCam != null ? mainCam : Camera.main;
        return mainCam != null ? mainCam.transform : transform;
    }

    private void SetVizActive(bool active)
    {
        if (vizObj == gameObject)
        {
            if (selfVizRenderers == null || selfVizRenderers.Length == 0)
            {
                selfVizRenderers = vizObj.GetComponentsInChildren<Renderer>(true);
            }

            for (int i = 0; i < selfVizRenderers.Length; i++)
            {
                if (selfVizRenderers[i] != null && selfVizRenderers[i].enabled != active)
                {
                    selfVizRenderers[i].enabled = active;
                }
            }

            return;
        }

        if (vizObj != null && vizObj.activeSelf != active)
        {
            vizObj.SetActive(active);
        }
    }
}
