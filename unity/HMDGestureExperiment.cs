using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class HMDGestureExperiment : MonoBehaviour
{
    [Header("References")]
    public Transform head;
    public TextMeshProUGUI instructionText;
    public Image gestureImage;

    [Header("Subject Settings")]
    public string subjectId = "subject_01";

    [Header("Experiment Settings")]
    public int totalTrials = 3;
    public int repetitionsPerGesture = 10;
    public float startupDelay = 5f;
    public float gestureDuration = 3f;
    public float restDuration = 2f;
    public float sampleInterval = 0.02f;

    [Header("Gesture Labels")]
    public List<string> gestureLabels = new List<string>
    {
        "Rotating_Left",
        "Rotating_Right",
        "Tilting_Up",
        "Tilting_Down",
        "Leaning_Left",
        "Leaning_Right"
    };

    [Header("Gesture Sprites")]
    public Sprite rotatingLeftSprite;
    public Sprite rotatingRightSprite;
    public Sprite tiltingUpSprite;
    public Sprite tiltingDownSprite;
    public Sprite leaningLeftSprite;
    public Sprite leaningRightSprite;

    private Vector3 previousEuler;
    private float previousSampleTime;

    private void Start()
    {
        if (head == null)
        {
            Debug.LogError("Head Transform is not assigned.");
            return;
        }

        if (instructionText == null)
        {
            Debug.LogError("Instruction Text is not assigned.");
            return;
        }

        Debug.Log("Save Directory: " + Application.persistentDataPath);

        previousEuler = GetNormalizedEuler(head.rotation.eulerAngles);
        previousSampleTime = Time.time;

        StartCoroutine(RunExperiment());
    }

    private IEnumerator RunExperiment()
    {
        HideGestureImage();

        ShowText(
            "<size=56>HMD GESTURE EXPERIMENT</size>\n\n" +
            "<size=38>Get Ready</size>",
            42f
        );

        yield return new WaitForSeconds(startupDelay);

        int gesturesPerTrial = gestureLabels.Count * repetitionsPerGesture;

        for (int trial = 1; trial <= totalTrials; trial++)
        {
            HideGestureImage();

            ShowText(
                "<size=56>TRIAL " + trial + " / " + totalTrials + "</size>\n\n" +
                "<size=38>Get Ready</size>",
                42f
            );

            yield return new WaitForSeconds(2f);

            string splitName = trial < totalTrials ? "train" : "test";
            string fileName = subjectId + "_trial_" + trial + "_" + splitName + ".csv";
            string filePath = Path.Combine(Application.persistentDataPath, fileName);

            List<string> trialGestureSequence = CreateGestureSequence();
            SaveHeader(filePath);

            Dictionary<string, int> gestureCounts = new Dictionary<string, int>();
            foreach (string label in gestureLabels)
            {
                gestureCounts[label] = 0;
            }

            for (int i = 0; i < trialGestureSequence.Count; i++)
            {
                string gestureLabel = trialGestureSequence[i];

                gestureCounts[gestureLabel]++;
                int currentGestureIndex = i + 1;
                int currentRepeatIndex = gestureCounts[gestureLabel];

                ShowGestureText(
                    trial,
                    currentGestureIndex,
                    gesturesPerTrial,
                    gestureLabel,
                    currentRepeatIndex
                );

                yield return new WaitForSeconds(0.5f);

                yield return StartCoroutine(RecordGesture(filePath, trial, gestureLabel));

                string nextGestureLabel = null;
                int nextRepeatIndex = 0;

                if (i + 1 < trialGestureSequence.Count)
                {
                    nextGestureLabel = trialGestureSequence[i + 1];

                    int countSoFar = 0;
                    for (int j = 0; j <= i; j++)
                    {
                        if (trialGestureSequence[j] == nextGestureLabel)
                        {
                            countSoFar++;
                        }
                    }

                    nextRepeatIndex = countSoFar + 1;
                }

                ShowRestPreview(
                    trial,
                    currentGestureIndex,
                    gesturesPerTrial,
                    nextGestureLabel,
                    nextRepeatIndex
                );

                yield return new WaitForSeconds(restDuration);
            }

            Debug.Log("CSV saved: " + filePath);

            HideGestureImage();

            ShowText(
                "<size=56>TRIAL " + trial + " SAVED</size>\n\n" +
                "<size=32>" + fileName + "</size>",
                40f
            );

            yield return new WaitForSeconds(2f);
        }

        ShowFinishedText();
    }

    private List<string> CreateGestureSequence()
    {
        List<string> sequence = new List<string>();

        foreach (string label in gestureLabels)
        {
            for (int i = 0; i < repetitionsPerGesture; i++)
            {
                sequence.Add(label);
            }
        }

        Shuffle(sequence);
        return sequence;
    }

    private void Shuffle(List<string> list)
    {
        for (int i = 0; i < list.Count; i++)
        {
            int randomIndex = UnityEngine.Random.Range(i, list.Count);
            string temp = list[i];
            list[i] = list[randomIndex];
            list[randomIndex] = temp;
        }
    }

    private IEnumerator RecordGesture(string filePath, int trial, string label)
    {
        float startTime = Time.time;
        float nextSampleTime = Time.time;

        previousEuler = GetNormalizedEuler(head.rotation.eulerAngles);
        previousSampleTime = Time.time;

        StringBuilder sb = new StringBuilder();

        while (Time.time - startTime < gestureDuration)
        {
            if (Time.time >= nextSampleTime)
            {
                float relativeTime = Time.time - startTime;

                Vector3 position = head.position;
                Quaternion rotation = head.rotation;
                Vector3 euler = GetNormalizedEuler(rotation.eulerAngles);

                float currentTime = Time.time;
                float deltaTime = Mathf.Max(currentTime - previousSampleTime, 0.0001f);

                Vector3 angularVelocity = CalculateAngularVelocity(previousEuler, euler, deltaTime);

                sb.Append(subjectId).Append(",");
                sb.Append(trial).Append(",");
                sb.Append(label).Append(",");
                sb.Append(relativeTime.ToString("F4")).Append(",");

                sb.Append(position.x.ToString("F6")).Append(",");
                sb.Append(position.y.ToString("F6")).Append(",");
                sb.Append(position.z.ToString("F6")).Append(",");

                sb.Append(rotation.x.ToString("F6")).Append(",");
                sb.Append(rotation.y.ToString("F6")).Append(",");
                sb.Append(rotation.z.ToString("F6")).Append(",");
                sb.Append(rotation.w.ToString("F6")).Append(",");

                sb.Append(euler.x.ToString("F6")).Append(",");
                sb.Append(euler.y.ToString("F6")).Append(",");
                sb.Append(euler.z.ToString("F6")).Append(",");

                sb.Append(angularVelocity.x.ToString("F6")).Append(",");
                sb.Append(angularVelocity.y.ToString("F6")).Append(",");
                sb.Append(angularVelocity.z.ToString("F6"));

                sb.AppendLine();

                previousEuler = euler;
                previousSampleTime = currentTime;

                nextSampleTime += sampleInterval;
            }

            yield return null;
        }

        File.AppendAllText(filePath, sb.ToString());
    }

    private void SaveHeader(string filePath)
    {
        string header =
            "subject_id,trial,label,timestamp," +
            "head_px,head_py,head_pz," +
            "head_rx,head_ry,head_rz,head_rw," +
            "head_ex,head_ey,head_ez," +
            "head_avx,head_avy,head_avz\n";

        File.WriteAllText(filePath, header);
    }

    private Vector3 GetNormalizedEuler(Vector3 euler)
    {
        return new Vector3(
            NormalizeAngle(euler.x),
            NormalizeAngle(euler.y),
            NormalizeAngle(euler.z)
        );
    }

    private float NormalizeAngle(float angle)
    {
        if (angle > 180f)
        {
            angle -= 360f;
        }

        return angle;
    }

    private Vector3 CalculateAngularVelocity(Vector3 previous, Vector3 current, float deltaTime)
    {
        float avx = Mathf.DeltaAngle(previous.x, current.x) / deltaTime;
        float avy = Mathf.DeltaAngle(previous.y, current.y) / deltaTime;
        float avz = Mathf.DeltaAngle(previous.z, current.z) / deltaTime;

        return new Vector3(avx, avy, avz);
    }

    private void ShowGestureText(
        int trial,
        int currentGestureIndex,
        int gesturesPerTrial,
        string gestureLabel,
        int currentRepeatIndex
    )
    {
        string gestureName = GetGestureName(gestureLabel);
        string gestureGuide = GetGestureGuide(gestureLabel);

        SetGestureImage(gestureLabel);

        string text =
            "<size=30>Trial " + trial + " / " + totalTrials +
            "     Gesture " + currentGestureIndex + " / " + gesturesPerTrial + "</size>\n" +
            "<size=30>Repeat " + currentRepeatIndex + " / " + repetitionsPerGesture + "</size>\n\n" +
            "<size=52>" + gestureName + "</size>\n\n" +
            "<size=32>" + gestureGuide + "</size>";

        ShowText(text, 40f);
    }

    private void ShowRestPreview(
        int trial,
        int currentGestureIndex,
        int gesturesPerTrial,
        string nextGestureLabel,
        int nextRepeatIndex
    )
    {
        string text;

        if (!string.IsNullOrEmpty(nextGestureLabel))
        {
            string nextGestureName = GetGestureName(nextGestureLabel);
            string nextGestureGuide = GetGestureGuide(nextGestureLabel);

            SetGestureImage(nextGestureLabel);

            text =
                "<size=30>Trial " + trial + " / " + totalTrials +
                "     Gesture " + currentGestureIndex + " / " + gesturesPerTrial + "</size>\n\n" +
                "<size=46>REST</size>\n\n" +
                "<size=28>Next Gesture</size>\n" +
                "<size=42>" + nextGestureName + "</size>\n" +
                "<size=28>Repeat " + nextRepeatIndex + " / " + repetitionsPerGesture + "</size>\n" +
                "<size=28>" + nextGestureGuide + "</size>";
        }
        else
        {
            HideGestureImage();

            text =
                "<size=30>Trial " + trial + " / " + totalTrials +
                "     Gesture " + currentGestureIndex + " / " + gesturesPerTrial + "</size>\n\n" +
                "<size=46>REST</size>\n\n" +
                "<size=34>Trial Completed</size>";
        }

        ShowText(text, 38f);
    }

    private void ShowFinishedText()
    {
        HideGestureImage();

        ShowText(
            "<size=56>FINISHED</size>\n\n" +
            "<size=36>Thank you</size>",
            42f
        );
    }

    private string GetGestureName(string label)
    {
        switch (label)
        {
            case "Rotating_Left":
                return "ROTATE LEFT";
            case "Rotating_Right":
                return "ROTATE RIGHT";
            case "Tilting_Up":
                return "TILT UP";
            case "Tilting_Down":
                return "TILT DOWN";
            case "Leaning_Left":
                return "LEAN LEFT";
            case "Leaning_Right":
                return "LEAN RIGHT";
            default:
                return label.Replace("_", " ").ToUpper();
        }
    }

    private string GetGestureGuide(string label)
    {
        switch (label)
        {
            case "Rotating_Left":
                return "Forward -> Left -> Forward";
            case "Rotating_Right":
                return "Forward -> Right -> Forward";
            case "Tilting_Up":
                return "Forward -> Up -> Forward";
            case "Tilting_Down":
                return "Forward -> Down -> Forward";
            case "Leaning_Left":
                return "Forward -> Lean Left -> Forward";
            case "Leaning_Right":
                return "Forward -> Lean Right -> Forward";
            default:
                return "Perform the gesture";
        }
    }

    private Sprite GetGestureSprite(string label)
    {
        switch (label)
        {
            case "Rotating_Left":
                return rotatingLeftSprite;
            case "Rotating_Right":
                return rotatingRightSprite;
            case "Tilting_Up":
                return tiltingUpSprite;
            case "Tilting_Down":
                return tiltingDownSprite;
            case "Leaning_Left":
                return leaningLeftSprite;
            case "Leaning_Right":
                return leaningRightSprite;
            default:
                return null;
        }
    }

    private void SetGestureImage(string label)
    {
        if (gestureImage == null) return;

        Sprite sprite = GetGestureSprite(label);

        if (sprite != null)
        {
            gestureImage.sprite = sprite;
            gestureImage.preserveAspect = true;
            gestureImage.enabled = true;
        }
        else
        {
            gestureImage.enabled = false;
        }
    }

    private void HideGestureImage()
    {
        if (gestureImage != null)
        {
            gestureImage.enabled = false;
        }
    }

    private void ShowText(string text, float baseFontSize)
    {
        if (instructionText == null) return;

        instructionText.text = text;
        instructionText.fontSize = baseFontSize;
        instructionText.alignment = TextAlignmentOptions.Center;
        instructionText.enableWordWrapping = true;
        instructionText.overflowMode = TextOverflowModes.Overflow;
        instructionText.richText = true;
    }
}