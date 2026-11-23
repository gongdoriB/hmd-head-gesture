using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.Collections.Generic;

public class HeadTracking : MonoBehaviour
{
    public Transform head;

    // Unity Inspector에서 Text(TMP)를 연결할 곳
    public TextMeshProUGUI resultText;

    public Image yesButtonImage;
    public Image noButtonImage;

    private Color normalYesColor;
    private Color normalNoColor;

    public float ignoreStartTime = 1.0f;
    public float detectCooldown = 1.0f;
    public float windowTime = 0.6f;

    public float nodPitchRangeThreshold = 18f;
    public float shakeYawRangeThreshold = 22f;

    private float lastDetectTime = 0f;

    private List<float> times = new List<float>();
    private List<float> pitches = new List<float>();
    private List<float> yaws = new List<float>();

    void Start()
    {
        if (head == null)
        {
            head = Camera.main.transform;
        }

        if (resultText != null)
        {
            resultText.text = "Continue?\nNOD = YES\nSHAKE = NO";
            normalYesColor = yesButtonImage.color;
            normalNoColor = noButtonImage.color;
        }

        Debug.Log("HeadTracking initialized");
    }

    void Update()
    {
        if (Time.time < ignoreStartTime)
        {
            return;
        }

        Vector3 euler = head.rotation.eulerAngles;

        float pitch = NormalizeAngle(euler.x);
        float yaw = NormalizeAngle(euler.y);
        float currentTime = Time.time;

        times.Add(currentTime);
        pitches.Add(pitch);
        yaws.Add(yaw);

        while (times.Count > 0 && currentTime - times[0] > windowTime)
        {
            times.RemoveAt(0);
            pitches.RemoveAt(0);
            yaws.RemoveAt(0);
        }

        if (currentTime - lastDetectTime < detectCooldown)
        {
            return;
        }

        float pitchRange = GetRange(pitches);
        float yawRange = GetRangeWithDeltaAngle(yaws);

        // 너무 큰 튐값은 무시
        if (yawRange > 120f || pitchRange > 120f)
        {
            ClearBuffer();
            return;
        }

        if (pitchRange > nodPitchRangeThreshold && pitchRange > yawRange)
        {
            Debug.Log($"NOD detected | Pitch Range: {pitchRange:F2}, Yaw Range: {yawRange:F2}");

            if (resultText != null)
            {
                resultText.text = "YES selected\n(NOD detected)";
                yesButtonImage.color = Color.green;
                noButtonImage.color = normalNoColor;
            }

            lastDetectTime = currentTime;
            ClearBuffer();
            return;
        }

        if (yawRange > shakeYawRangeThreshold && yawRange > pitchRange)
        {
            Debug.Log($"SHAKE detected | Yaw Range: {yawRange:F2}, Pitch Range: {pitchRange:F2}");

            if (resultText != null)
            {
                resultText.text = "NO selected\n(SHAKE detected)";
                noButtonImage.color = Color.red;
                yesButtonImage.color = normalYesColor;
            }

            lastDetectTime = currentTime;
            ClearBuffer();
            return;
        }
    }

    float GetRange(List<float> values)
    {
        if (values.Count == 0) return 0f;

        float min = values[0];
        float max = values[0];

        foreach (float v in values)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        return max - min;
    }

    float GetRangeWithDeltaAngle(List<float> values)
    {
        if (values.Count == 0) return 0f;

        float baseAngle = values[0];
        float minDelta = 0f;
        float maxDelta = 0f;

        foreach (float v in values)
        {
            float delta = Mathf.DeltaAngle(baseAngle, v);

            if (delta < minDelta) minDelta = delta;
            if (delta > maxDelta) maxDelta = delta;
        }

        return maxDelta - minDelta;
    }

    void ClearBuffer()
    {
        times.Clear();
        pitches.Clear();
        yaws.Clear();
    }

    float NormalizeAngle(float angle)
    {
        if (angle > 180f)
        {
            angle -= 360f;
        }

        return angle;
    }
}