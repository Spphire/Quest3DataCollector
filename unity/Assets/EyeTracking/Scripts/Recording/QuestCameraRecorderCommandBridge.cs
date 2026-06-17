using System;
using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;

namespace EyeTracking.Recording
{
    [DisallowMultipleComponent]
    public sealed class QuestCameraRecorderCommandBridge : MonoBehaviour
    {
        [SerializeField] private QuestCameraRecorder recorder;
        [SerializeField] private QuestPcCalibrationRecorder calibrationRecorder;
        [SerializeField] private bool enableFileCommands = true;
        [SerializeField] private string commandFileName = "record_command.txt";
        [SerializeField, Min(0.1f)] private float pollIntervalSeconds = 0.25f;
        [SerializeField, Min(0.1f)] private float startReadyTimeoutSeconds = 30f;
        [SerializeField] private bool deleteCommandAfterRead = true;
        [SerializeField] private bool logCommandStatus = true;

        private string commandPath;
        private float nextPollTime;
        private DateTime lastCommandWriteTimeUtc;
        private Coroutine pendingStart;

        public string CommandPath => commandPath;
        public bool EnableFileCommands
        {
            get { return enableFileCommands; }
            set { enableFileCommands = value; }
        }

        private void Awake()
        {
            ResolveReferences();
            commandPath = Path.Combine(Application.persistentDataPath, commandFileName);
            if (logCommandStatus)
            {
                Debug.Log("[QuestCameraRecorderCommandBridge] File command path: " + commandPath, this);
            }
        }

        private void Update()
        {
            if (!enableFileCommands || Time.unscaledTime < nextPollTime)
            {
                return;
            }

            nextPollTime = Time.unscaledTime + pollIntervalSeconds;
            PollCommandFile();
        }

        public void RequestStartRecording()
        {
            ResolveReferences();
            if (recorder == null)
            {
                Debug.LogWarning("[QuestCameraRecorderCommandBridge] Cannot start: recorder reference is missing.", this);
                return;
            }

            if (recorder.IsRecording)
            {
                LogStatus("start ignored; recorder is already recording.");
                return;
            }

            if (pendingStart != null)
            {
                LogStatus("start ignored; a pending start is already waiting for recorder readiness.");
                return;
            }

            pendingStart = StartCoroutine(StartWhenReady());
        }

        public void RequestStopRecording()
        {
            ResolveReferences();
            if (pendingStart != null)
            {
                StopCoroutine(pendingStart);
                pendingStart = null;
            }

            if (recorder == null)
            {
                Debug.LogWarning("[QuestCameraRecorderCommandBridge] Cannot stop: recorder reference is missing.", this);
                return;
            }

            recorder.StopRecording();
            LogStatus("stop requested.");
        }

        public void RequestStartCalibrationRecording()
        {
            ResolveReferences();
            if (calibrationRecorder == null)
            {
                Debug.LogWarning("[QuestCameraRecorderCommandBridge] Cannot start calibration: calibration recorder reference is missing.", this);
                return;
            }

            calibrationRecorder.StartCalibrationRecording();
            LogStatus("calibration start requested.");
        }

        public void RequestStopCalibrationRecording()
        {
            ResolveReferences();
            if (calibrationRecorder == null)
            {
                Debug.LogWarning("[QuestCameraRecorderCommandBridge] Cannot stop calibration: calibration recorder reference is missing.", this);
                return;
            }

            calibrationRecorder.StopCalibrationRecording();
            LogStatus("calibration stop requested.");
        }

        public void RequestToggleCalibrationRecording()
        {
            ResolveReferences();
            if (calibrationRecorder == null)
            {
                Debug.LogWarning("[QuestCameraRecorderCommandBridge] Cannot toggle calibration: calibration recorder reference is missing.", this);
                return;
            }

            calibrationRecorder.ToggleCalibrationRecording();
            LogStatus("calibration toggle requested.");
        }

        private IEnumerator StartWhenReady()
        {
            float deadline = Time.realtimeSinceStartup + startReadyTimeoutSeconds;
            while (recorder != null && !recorder.IsReady && Time.realtimeSinceStartup < deadline)
            {
                yield return null;
            }

            pendingStart = null;
            if (recorder == null)
            {
                Debug.LogWarning("[QuestCameraRecorderCommandBridge] Timed out waiting to start: recorder reference disappeared.", this);
                yield break;
            }

            if (!recorder.IsReady)
            {
                Debug.LogWarning(
                    "[QuestCameraRecorderCommandBridge] Timed out waiting to start: " +
                    $"status={recorder.LastInitStatus} error={recorder.LastInitError}",
                    this);
                yield break;
            }

            recorder.StartRecording();
            LogStatus("start requested.");
        }

        private void PollCommandFile()
        {
            if (string.IsNullOrEmpty(commandPath) || !File.Exists(commandPath))
            {
                return;
            }

            string commandText;
            DateTime writeTimeUtc;
            try
            {
                writeTimeUtc = File.GetLastWriteTimeUtc(commandPath);
                if (!deleteCommandAfterRead && writeTimeUtc <= lastCommandWriteTimeUtc)
                {
                    return;
                }

                commandText = File.ReadAllText(commandPath, Encoding.UTF8).Trim();
                if (deleteCommandAfterRead)
                {
                    File.Delete(commandPath);
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning("[QuestCameraRecorderCommandBridge] Could not read command file: " + exception.Message, this);
                return;
            }

            if (string.IsNullOrEmpty(commandText))
            {
                return;
            }

            lastCommandWriteTimeUtc = writeTimeUtc;
            ExecuteCommand(commandText);
        }

        private void ExecuteCommand(string commandText)
        {
            string command = commandText.Split(new[] { '\r', '\n', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)[0]
                .Trim()
                .ToLowerInvariant();

            if (command == "start")
            {
                RequestStartRecording();
                return;
            }

            if (command == "stop")
            {
                RequestStopRecording();
                return;
            }

            if (command == "calib_start" || command == "calibration_start" || command == "start_calibration")
            {
                RequestStartCalibrationRecording();
                return;
            }

            if (command == "calib_stop" || command == "calibration_stop" || command == "stop_calibration")
            {
                RequestStopCalibrationRecording();
                return;
            }

            if (command == "calib_toggle" || command == "calibration_toggle" || command == "toggle_calibration")
            {
                RequestToggleCalibrationRecording();
                return;
            }

            Debug.LogWarning("[QuestCameraRecorderCommandBridge] Unknown command: " + commandText, this);
        }

        private void ResolveReferences()
        {
            if (recorder == null)
            {
                recorder = GetComponent<QuestCameraRecorder>();
                if (recorder == null)
                {
                    recorder = FindFirstObjectByType<QuestCameraRecorder>();
                }
            }

            if (calibrationRecorder == null)
            {
                calibrationRecorder = GetComponent<QuestPcCalibrationRecorder>();
                if (calibrationRecorder == null)
                {
                    calibrationRecorder = FindFirstObjectByType<QuestPcCalibrationRecorder>();
                }
            }
        }

        private void LogStatus(string message)
        {
            if (logCommandStatus)
            {
                Debug.Log("[QuestCameraRecorderCommandBridge] " + message, this);
            }
        }
    }
}
