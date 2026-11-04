import csv
import io
import json
import os
import time
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3

s3 = boto3.client("s3")
sns = boto3.client("sns")

SNS_TOPIC_ARN = os.environ.get(
    "SNS_TOPIC_ARN",
    "arn:aws:sns:ap-southeast-2:192980391398:settlenet-anomaly-alerts"
)

BUCKET_NAME = os.environ.get("BUCKET_NAME", "settlenet-dev-apar-7")
MODEL_PREFIX = "models/"
INCOMING_PREFIX = "incoming/"
PREDICTIONS_PREFIX = "predictions/"


def send_anomaly_alert(source_key, anomalies):
    anomaly_count = len(anomalies)

    message = (
        "SettleNet anomaly alert\n\n"
        f"Source file: {source_key}\n"
        f"Anomalies detected: {anomaly_count}\n\n"
        "Flagged transactions:\n"
    )

    for anomaly in anomalies:
        message += (
            f"- {anomaly['transaction_id']}: "
            f"₹{anomaly['amount']} "
            f"(modified z-score: {anomaly['modified_z_score']})\n"
        )

    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="SettleNet: Transaction Anomaly Detected",
        Message=message
    )


def lambda_handler(event, context):
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        source_key = unquote_plus(
            record["s3"]["object"]["key"],
            encoding="utf-8"
        )

        if bucket != BUCKET_NAME or not source_key.startswith(INCOMING_PREFIX):
            continue

        result = score_transactions(bucket, source_key)
        results.append(result)

    return {
        "statusCode": 200,
        "inference_results": results
    }


def score_transactions(bucket, source_key):
    start_time = time.time()

    try:
        model = get_latest_model(bucket)

        if not model:
            return {
                "source_object": f"s3://{bucket}/{source_key}",
                "inference_status": "SKIPPED",
                "reason": "No trained model found",
                "inference_time_ms": round(
                    (time.time() - start_time) * 1000, 2
                )
            }

        source_object = s3.get_object(Bucket=bucket, Key=source_key)
        content = source_object["Body"].read().decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(content)))

        parameters = model["parameters"]
        median_amount = parameters["median_amount"]
        mad = parameters["median_absolute_deviation"]
        lower_threshold = parameters["lower_threshold"]
        upper_threshold = parameters["upper_threshold"]

        predictions = []

        for row in rows:
            try:
                amount = float(row["amount"])
            except ValueError:
                predictions.append({
                    "transaction_id": row.get("transaction_id"),
                    "amount": row.get("amount"),
                    "prediction": "UNSCORED",
                    "reason": "Amount is not numeric"
                })
                continue

            modified_z_score = (
                0
                if mad == 0
                else 0.6745 * (amount - median_amount) / mad
            )

            is_anomaly = (
                amount > upper_threshold or amount < lower_threshold
            )

            predictions.append({
                "transaction_id": row.get("transaction_id"),
                "customer_id": row.get("customer_id"),
                "amount": amount,
                "modified_z_score": round(modified_z_score, 3),
                "prediction": "ANOMALY" if is_anomaly else "NORMAL",
                "normal_range": {
                    "lower": lower_threshold,
                    "upper": upper_threshold
                }
            })

        anomaly_count = sum(
            1 for prediction in predictions
            if prediction["prediction"] == "ANOMALY"
        )

        inference_time_ms = round(
            (time.time() - start_time) * 1000, 2
        )

        inference_result = {
            "inference_id": datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            ),
            "model_id": model["model_id"],
            "model_version": model["model_version"],
            "source_object": f"s3://{bucket}/{source_key}",
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows_scored": len(predictions),
            "anomalies_detected": anomaly_count,
            "inference_time_ms": inference_time_ms,
            "model_parameters": parameters,
            "predictions": predictions
        }

        file_name = os.path.basename(source_key)
        prediction_key = (
            f"{PREDICTIONS_PREFIX}"
            f"{inference_result['inference_id']}_{file_name}_predictions.json"
        )

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=prediction_key,
            Body=json.dumps(inference_result, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

        anomalies = [
            prediction for prediction in predictions
            if prediction["prediction"] == "ANOMALY"
        ]

        if anomalies:
            send_anomaly_alert(source_key, anomalies)

        return {
            "inference_status": "COMPLETED",
            "source_object": f"s3://{bucket}/{source_key}",
            "prediction_report": f"s3://{bucket}/{prediction_key}",
            "rows_scored": len(predictions),
            "anomalies_detected": anomaly_count,
            "inference_time_ms": inference_time_ms
        }

    except Exception as error:
        return {
            "source_object": f"s3://{bucket}/{source_key}",
            "inference_status": "ERROR",
            "error": str(error),
            "inference_time_ms": round(
                (time.time() - start_time) * 1000, 2
            )
        }


def get_latest_model(bucket):
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=MODEL_PREFIX
    )

    model_objects = [
        item for item in response.get("Contents", [])
        if item["Key"].endswith("_anomaly_model.json")
    ]

    if not model_objects:
        return None

    latest = max(
        model_objects,
        key=lambda item: item["LastModified"]
    )

    model_object = s3.get_object(Bucket=bucket, Key=latest["Key"])

    return json.loads(
        model_object["Body"].read().decode("utf-8")
    )
