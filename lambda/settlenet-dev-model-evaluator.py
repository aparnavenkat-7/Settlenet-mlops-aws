import json
import csv
import io
import os
import boto3

s3 = boto3.client("s3")

BUCKET = os.environ["S3_BUCKET"]
TEST_KEY = "evaluation/test_labelled.csv"
METRICS_KEY = "evaluation/metrics.json"

ANOMALY_THRESHOLD = 1000


def predict_label(amount):
    return 1 if amount > ANOMALY_THRESHOLD else 0


def lambda_handler(event, context):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=TEST_KEY)
        csv_content = obj["Body"].read().decode("utf-8")

        reader = csv.DictReader(io.StringIO(csv_content))

        y_true = []
        y_pred = []

        for row in reader:
            amount = float(row["amount"])
            true_label = int(row["true_label"])

            y_true.append(true_label)
            y_pred.append(predict_label(amount))

        total = len(y_true)

        if total == 0:
            raise ValueError("The test CSV file is empty.")

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "total_records": total,
            "normal_records": y_true.count(0),
            "anomaly_records": y_true.count(1),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            }
        }

        s3.put_object(
            Bucket=BUCKET,
            Key=METRICS_KEY,
            Body=json.dumps(metrics, indent=2),
            ContentType="application/json"
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Model evaluation completed",
                "metrics_file": f"s3://{BUCKET}/{METRICS_KEY}",
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }, indent=2)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
