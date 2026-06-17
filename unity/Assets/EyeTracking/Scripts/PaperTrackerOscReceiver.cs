using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Events;

namespace EyeTracking
{
    public sealed class PaperTrackerOscReceiver : MonoBehaviour
    {
        [Serializable]
        public sealed class Vector2Event : UnityEvent<Vector2> { }

        [Serializable]
        public sealed class FloatEvent : UnityEvent<float> { }

        public struct ScreenDebugDot
        {
            public Vector2 ViewportPosition;
            public float DiameterPixels;
            public Color Color;

            public ScreenDebugDot(Vector2 viewportPosition, float diameterPixels, Color color)
            {
                ViewportPosition = viewportPosition;
                DiameterPixels = diameterPixels;
                Color = color;
            }
        }

        private const string CenterPitchYawAddress = "/tracking/eye/CenterPitchYaw";
        private const string EyesClosedAmountAddress = "/tracking/eye/EyesClosedAmount";

        [Header("OSC")]
        [SerializeField] private int port = 9000;
        [SerializeField] private bool listenOnAnyAddress = true;
        [SerializeField] private string pcIp = "10.128.0.227";
        [SerializeField] private bool filterByPcIp = true;
        [SerializeField] private bool logMessages = true;
        [SerializeField] private int maxMessagesPerFrame = 64;
        [SerializeField, Min(0.05f)] private float connectionTimeout = 0.5f;

        [Header("Gaze Debug")]
        [SerializeField] private Transform rayOrigin;
        [SerializeField] private Transform gazeTarget;
        [SerializeField] private bool updateGazeTarget;
        [SerializeField] private float gazeTargetDistance = 2f;
        [SerializeField] private bool drawScreenDebugDot = true;
        [SerializeField, Min(0.1f)] private float screenDebugDotDistance = 2f;
        [SerializeField, Min(1f)] private float screenDebugDotDiameter = 48f;
        [SerializeField] private Color screenDebugDotColor = Color.red;
        [SerializeField] private bool drawCenterDotWhenDisconnected = true;
        [SerializeField, Range(0.05f, 1f)] private float disconnectedDotAlpha = 0.5f;

        [Header("Latest Values")]
        [SerializeField] private bool hasGaze;
        [SerializeField] private Vector2 centerPitchYaw;
        [SerializeField] private Vector3 centerDirectionLocal = Vector3.forward;
        [SerializeField] private float eyesClosedAmount;
        [SerializeField] private string lastRemoteEndpoint;
        [SerializeField] private float secondsSinceLastPacket = -1f;

        public Vector2Event CenterPitchYawReceived = new Vector2Event();
        public FloatEvent EyesClosedAmountReceived = new FloatEvent();

        private static readonly List<PaperTrackerOscReceiver> ActiveReceivers = new List<PaperTrackerOscReceiver>();

        private readonly ConcurrentQueue<OscMessage> messageQueue = new ConcurrentQueue<OscMessage>();
        private readonly ConcurrentQueue<string> logQueue = new ConcurrentQueue<string>();

        private UdpClient udpClient;
        private Thread receiveThread;
        private volatile bool running;
        private bool loggedIgnoredSender;
        private float lastPacketTime = -1f;

        public bool HasGaze
        {
            get { return hasGaze; }
        }

        public Vector2 CenterPitchYaw
        {
            get { return centerPitchYaw; }
        }

        public Vector3 CenterDirectionLocal
        {
            get { return centerDirectionLocal; }
        }

        public float EyesClosedAmount
        {
            get { return eyesClosedAmount; }
        }

        public float SecondsSinceLastPacket
        {
            get { return lastPacketTime >= 0f ? Time.time - lastPacketTime : -1f; }
        }

        public bool IsConnected
        {
            get { return running && hasGaze && SecondsSinceLastPacket >= 0f && SecondsSinceLastPacket <= connectionTimeout; }
        }

        public Ray WorldGazeRay
        {
            get
            {
                Transform origin = rayOrigin != null ? rayOrigin : transform;
                return new Ray(origin.position, origin.TransformDirection(centerDirectionLocal));
            }
        }

        private void OnEnable()
        {
            if (!ActiveReceivers.Contains(this))
            {
                ActiveReceivers.Add(this);
            }

            StartListening();
        }

        private void OnDisable()
        {
            ActiveReceivers.Remove(this);
            StopListening();
        }

        private void OnApplicationQuit()
        {
            StopListening();
        }

        private void Update()
        {
            DrainLogs();

            int processed = 0;
            while (processed < maxMessagesPerFrame && messageQueue.TryDequeue(out OscMessage message))
            {
                processed++;
                HandleMessage(message);
            }

            secondsSinceLastPacket = SecondsSinceLastPacket;

            if (updateGazeTarget && gazeTarget != null && hasGaze)
            {
                Ray ray = WorldGazeRay;
                gazeTarget.position = ray.origin + ray.direction * gazeTargetDistance;
                gazeTarget.rotation = Quaternion.LookRotation(ray.direction, Vector3.up);
            }
        }

        private void OnDrawGizmosSelected()
        {
            if (!hasGaze)
            {
                return;
            }

            Ray ray = WorldGazeRay;
            Gizmos.color = Color.green;
            Gizmos.DrawRay(ray.origin, ray.direction * gazeTargetDistance);
        }

        public static bool TryGetScreenDebugDot(Camera camera, out ScreenDebugDot dot)
        {
            dot = default(ScreenDebugDot);

            if (camera == null)
            {
                return false;
            }

            for (int i = ActiveReceivers.Count - 1; i >= 0; i--)
            {
                PaperTrackerOscReceiver receiver = ActiveReceivers[i];
                if (receiver == null)
                {
                    ActiveReceivers.RemoveAt(i);
                    continue;
                }

                if (receiver.TryGetScreenDebugDotForCamera(camera, out dot))
                {
                    return true;
                }
            }

            return false;
        }

        private bool TryGetScreenDebugDotForCamera(Camera camera, out ScreenDebugDot dot)
        {
            dot = default(ScreenDebugDot);

            if (!drawScreenDebugDot)
            {
                return false;
            }

            if (!IsConnected)
            {
                if (!drawCenterDotWhenDisconnected)
                {
                    return false;
                }

                Color color = screenDebugDotColor;
                color.a *= disconnectedDotAlpha;
                dot = new ScreenDebugDot(new Vector2(0.5f, 0.5f), screenDebugDotDiameter, color);
                return true;
            }

            Ray ray = GetCameraGazeRay(camera);
            Vector3 viewportPoint = camera.WorldToViewportPoint(ray.origin + ray.direction * screenDebugDotDistance);
            if (viewportPoint.z <= 0f)
            {
                return false;
            }

            dot = new ScreenDebugDot(
                new Vector2(Mathf.Clamp01(viewportPoint.x), 1f - Mathf.Clamp01(viewportPoint.y)),
                screenDebugDotDiameter,
                screenDebugDotColor
            );
            return true;
        }

        private Ray GetCameraGazeRay(Camera camera)
        {
            Transform origin = rayOrigin != null ? rayOrigin : camera.transform;
            return new Ray(origin.position, camera.transform.TransformDirection(centerDirectionLocal));
        }

        private void StartListening()
        {
            if (running)
            {
                return;
            }

            try
            {
                IPAddress address = listenOnAnyAddress ? IPAddress.Any : IPAddress.Loopback;
                udpClient = new UdpClient(new IPEndPoint(address, port));
                udpClient.Client.ReceiveTimeout = 100;

                running = true;
                receiveThread = new Thread(ReceiveLoop);
                receiveThread.IsBackground = true;
                receiveThread.Name = "PaperTracker OSC Receiver";
                receiveThread.Start();

                string senderFilter = filterByPcIp && !string.IsNullOrWhiteSpace(pcIp)
                    ? ", accepting sender " + pcIp
                    : ", accepting any sender";
                Debug.Log("PaperTracker OSC receiver listening on UDP " + address + ":" + port + senderFilter, this);
            }
            catch (Exception exception)
            {
                running = false;
                Debug.LogError("Failed to start PaperTracker OSC receiver on UDP port " + port + ": " + exception.Message, this);
                StopListening();
            }
        }

        private void StopListening()
        {
            running = false;

            if (udpClient != null)
            {
                udpClient.Close();
                udpClient = null;
            }

            if (receiveThread != null)
            {
                receiveThread.Join(200);
                receiveThread = null;
            }
        }

        private void ReceiveLoop()
        {
            IPEndPoint remoteEndpoint = new IPEndPoint(IPAddress.Any, 0);

            while (running)
            {
                try
                {
                    byte[] bytes = udpClient.Receive(ref remoteEndpoint);
                    if (!IsExpectedSender(remoteEndpoint.Address))
                    {
                        continue;
                    }

                    List<OscMessage> messages = new List<OscMessage>();
                    ParseOscPacket(bytes, remoteEndpoint.ToString(), messages);

                    for (int i = 0; i < messages.Count; i++)
                    {
                        messageQueue.Enqueue(messages[i]);
                    }
                }
                catch (SocketException exception)
                {
                    if (exception.SocketErrorCode != SocketError.TimedOut && running)
                    {
                        logQueue.Enqueue("PaperTracker OSC socket error: " + exception.Message);
                    }
                }
                catch (ObjectDisposedException)
                {
                    return;
                }
                catch (Exception exception)
                {
                    if (running)
                    {
                        logQueue.Enqueue("PaperTracker OSC parse error: " + exception.Message);
                    }
                }
            }
        }

        private bool IsExpectedSender(IPAddress remoteAddress)
        {
            if (!filterByPcIp || string.IsNullOrWhiteSpace(pcIp))
            {
                return true;
            }

            if (remoteAddress.ToString() == pcIp.Trim())
            {
                return true;
            }

            if (!loggedIgnoredSender)
            {
                loggedIgnoredSender = true;
                logQueue.Enqueue("Ignoring OSC packet from " + remoteAddress + ". Expected PC IP is " + pcIp + ".");
            }

            return false;
        }

        private void DrainLogs()
        {
            while (logQueue.TryDequeue(out string line))
            {
                Debug.LogWarning(line, this);
            }
        }

        private void HandleMessage(OscMessage message)
        {
            lastPacketTime = Time.time;
            lastRemoteEndpoint = message.RemoteEndpoint;

            if (logMessages)
            {
                Debug.Log(message.ToLogString(), this);
            }

            if (message.Address == CenterPitchYawAddress && message.Arguments.Length >= 2)
            {
                if (TryGetFloat(message.Arguments[0], out float pitch) &&
                    TryGetFloat(message.Arguments[1], out float yaw))
                {
                    hasGaze = true;
                    centerPitchYaw = new Vector2(pitch, yaw);
                    centerDirectionLocal = PitchYawToDirection(pitch, yaw);
                    CenterPitchYawReceived.Invoke(centerPitchYaw);
                }
            }
            else if (message.Address == EyesClosedAmountAddress && message.Arguments.Length >= 1)
            {
                if (TryGetFloat(message.Arguments[0], out float closedAmount))
                {
                    eyesClosedAmount = closedAmount;
                    EyesClosedAmountReceived.Invoke(eyesClosedAmount);
                }
            }
        }

        private static Vector3 PitchYawToDirection(float pitchDegrees, float yawDegrees)
        {
            return (Quaternion.Euler(pitchDegrees, yawDegrees, 0f) * Vector3.forward).normalized;
        }

        private static bool TryGetFloat(object value, out float result)
        {
            if (value is float floatValue)
            {
                result = floatValue;
                return true;
            }

            if (value is int intValue)
            {
                result = intValue;
                return true;
            }

            if (value is double doubleValue)
            {
                result = (float)doubleValue;
                return true;
            }

            result = 0f;
            return false;
        }

        private static void ParseOscPacket(byte[] bytes, string remoteEndpoint, List<OscMessage> messages)
        {
            int offset = 0;
            string head = ReadOscString(bytes, ref offset);

            if (head == "#bundle")
            {
                ReadInt64(bytes, ref offset);

                while (offset < bytes.Length)
                {
                    int elementSize = ReadInt32(bytes, ref offset);
                    if (elementSize <= 0 || offset + elementSize > bytes.Length)
                    {
                        throw new FormatException("Invalid OSC bundle element size: " + elementSize);
                    }

                    byte[] element = new byte[elementSize];
                    Array.Copy(bytes, offset, element, 0, elementSize);
                    offset += elementSize;
                    ParseOscPacket(element, remoteEndpoint, messages);
                }

                return;
            }

            if (!head.StartsWith("/", StringComparison.Ordinal))
            {
                throw new FormatException("OSC packet does not start with an address. First string: " + head);
            }

            string typeTags = ReadOscString(bytes, ref offset);
            if (!typeTags.StartsWith(",", StringComparison.Ordinal))
            {
                throw new FormatException("OSC message missing type tags for address " + head);
            }

            List<object> arguments = new List<object>();
            for (int i = 1; i < typeTags.Length; i++)
            {
                char tag = typeTags[i];
                switch (tag)
                {
                    case 'i':
                        arguments.Add(ReadInt32(bytes, ref offset));
                        break;
                    case 'h':
                        arguments.Add(ReadInt64(bytes, ref offset));
                        break;
                    case 'f':
                        arguments.Add(ReadFloat32(bytes, ref offset));
                        break;
                    case 'd':
                        arguments.Add(ReadFloat64(bytes, ref offset));
                        break;
                    case 's':
                        arguments.Add(ReadOscString(bytes, ref offset));
                        break;
                    case 'b':
                        arguments.Add(ReadBlob(bytes, ref offset));
                        break;
                    case 'T':
                        arguments.Add(true);
                        break;
                    case 'F':
                        arguments.Add(false);
                        break;
                    case 'N':
                        arguments.Add(null);
                        break;
                    case 'I':
                        arguments.Add("Impulse");
                        break;
                    default:
                        throw new NotSupportedException("Unsupported OSC argument type tag: " + tag);
                }
            }

            messages.Add(new OscMessage(head, typeTags, arguments.ToArray(), remoteEndpoint));
        }

        private static string ReadOscString(byte[] bytes, ref int offset)
        {
            int start = offset;
            while (offset < bytes.Length && bytes[offset] != 0)
            {
                offset++;
            }

            if (offset >= bytes.Length)
            {
                throw new FormatException("Unterminated OSC string.");
            }

            string value = Encoding.UTF8.GetString(bytes, start, offset - start);
            offset++;

            while ((offset - start) % 4 != 0)
            {
                offset++;
                if (offset > bytes.Length)
                {
                    throw new FormatException("OSC string padding extends beyond packet.");
                }
            }

            return value;
        }

        private static byte[] ReadBytes(byte[] bytes, ref int offset, int count)
        {
            if (count < 0 || offset + count > bytes.Length)
            {
                throw new FormatException("OSC packet ended while reading " + count + " bytes.");
            }

            byte[] value = new byte[count];
            Array.Copy(bytes, offset, value, 0, count);
            offset += count;
            return value;
        }

        private static int ReadInt32(byte[] bytes, ref int offset)
        {
            byte[] value = ReadBytes(bytes, ref offset, 4);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(value);
            }

            return BitConverter.ToInt32(value, 0);
        }

        private static long ReadInt64(byte[] bytes, ref int offset)
        {
            byte[] value = ReadBytes(bytes, ref offset, 8);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(value);
            }

            return BitConverter.ToInt64(value, 0);
        }

        private static float ReadFloat32(byte[] bytes, ref int offset)
        {
            byte[] value = ReadBytes(bytes, ref offset, 4);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(value);
            }

            return BitConverter.ToSingle(value, 0);
        }

        private static double ReadFloat64(byte[] bytes, ref int offset)
        {
            byte[] value = ReadBytes(bytes, ref offset, 8);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(value);
            }

            return BitConverter.ToDouble(value, 0);
        }

        private static byte[] ReadBlob(byte[] bytes, ref int offset)
        {
            int length = ReadInt32(bytes, ref offset);
            byte[] blob = ReadBytes(bytes, ref offset, length);

            while (offset % 4 != 0)
            {
                offset++;
                if (offset > bytes.Length)
                {
                    throw new FormatException("OSC blob padding extends beyond packet.");
                }
            }

            return blob;
        }

        private sealed class OscMessage
        {
            public readonly string Address;
            public readonly string TypeTags;
            public readonly object[] Arguments;
            public readonly string RemoteEndpoint;

            public OscMessage(string address, string typeTags, object[] arguments, string remoteEndpoint)
            {
                Address = address;
                TypeTags = typeTags;
                Arguments = arguments;
                RemoteEndpoint = remoteEndpoint;
            }

            public string ToLogString()
            {
                string[] formatted = new string[Arguments.Length];
                for (int i = 0; i < Arguments.Length; i++)
                {
                    formatted[i] = FormatArgument(Arguments[i]);
                }

                return "OSC " + Address + " " + TypeTags + " [" + string.Join(", ", formatted) + "] from " + RemoteEndpoint;
            }

            private static string FormatArgument(object value)
            {
                if (value == null)
                {
                    return "null";
                }

                if (value is float floatValue)
                {
                    return floatValue.ToString("0.#####", CultureInfo.InvariantCulture);
                }

                if (value is double doubleValue)
                {
                    return doubleValue.ToString("0.#####", CultureInfo.InvariantCulture);
                }

                byte[] blob = value as byte[];
                if (blob != null)
                {
                    return "blob[" + blob.Length + "]";
                }

                return value.ToString();
            }
        }
    }
}
