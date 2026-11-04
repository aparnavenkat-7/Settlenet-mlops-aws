import csv
import io
import json
import os
import statistics
import uuid
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3

s3 = boto3.client("s3")

BUCKET_NAME = os.environ.get("BUCKET_NAME", "settlenet-dev-apar-7")
REPORT_PREFIX = "validation-reports/"
MODEL_PREFIX = "models/"
EXPERIMENT_PREFIX = "experiments/"


def lambda_handler(event, context):
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        report_key = unquote_plus(record["s3"]["object"]["key"], encoding="utf-8")

        if bucket != BUCKET_NAME or not report_key.startswith(REPORT_PREFIX):
            continue

        result = train_from_validation_report(bucket, report_key)
        results.append(result)

    return {
        "statusCode": 200,
        "training_results": results
    }


def train_from_validation_report(bucket, report_key):
    try:
        report_object = s3.get_object(Bucket=bucket, Key=report_key)
        report = json.loads(report_object["Body"].read().decode("utf-8"))

        if report.get("status") != "PASSED":
            return {
                "validation_report": f"s3://{bucket}/{report_key}",
                "training_status": "SKIPPED",
                "reason": "Validation status is not PASSED"
            }

        source_key = report["source_object"].replace(f"s3://{bucket}/", "")
        source_object = s3.get_object(Bucket=bucket, Key=source_key)
        content = source_object["Body"].read().decode("utf-8-sig")

        rows = list(csv.DictReader(io.StringIO(content)))

        if len(rows) < 3:
            return {
                "validation_report": f"s3://{bucket}/{report_key}",
                "training_status": "SKIPPED",
                "reason": "At least 3 valid rows are required for training"
            }

        amounts = []
        for row in rows:
            try:
                amount = float(row["amount"])
                if amount > 0:
                    amounts.append(amount)
            except ValueError:
                continue

        if len(amounts) < 3:
            return {
                "validation_report": f"s3://{bucket}/{report_key}",
                "training_status": "SKIPPED",
                "reason": "At least 3 numeric positive amounts are required"
            }

        median_amount = statistics.median(amounts)
        absolute_deviations = [abs(amount - median_amount) for amount in amounts]
        mad = statistics.median(absolute_deviations)

        if mad == 0:
            upper_threshold = median_amount * 3
            lower_threshold = 0
        else:
            upper_threshold = median_amount + (3.5 * mad / 0.6745)
            lower_threshold = max(0, median_amount - (3.5 * mad / 0.6745))

        anomalies = []
        for row in rows:
            try:
                amount = float(row["amount"])
            except ValueError:
                continue

            is_anomaly = amount > upper_threshold or amount < lower_threshold
            modified_z_score = (
                0
                if mad == 0
                else 0.6745 * (amount - median_amount) / mad
            )

            if is_anomaly:
                anomalies.append({
                    "transaction_id": row.get("transaction_id"),
                    "amount": amount,
                    "modified_z_score": round(modified_z_score, 3),
                    "reason": "Amount outside robust median/MAD threshold range"
                })

        model_id = str(uuid.uuid4())
        trained_at = datetime.now(timezone.utc).isoformat()

        model_artifact = {
            "model_id": model_id,
            "model_name": "settlenet-transaction-anomaly-detector",
            "model_version": "1.1.0",
            "algorithm": "median_mad_amount_anomaly_detection",
            "trained_at_utc": trained_at,
            "training_data": f"s3://{bucket}/{source_key}",
            "validation_report": f"s3://{bucket}/{report_key}",
            "rows_used": len(amounts),
            "parameters": {
                "median_amount": round(median_amount, 4),
                "median_absolute_deviation": round(mad, 4),
                "lower_threshold": round(lower_threshold, 4),
                "upper_threshold": round(upper_threshold, 4),
                "modified_z_score_cutoff": 3.5
            }
        }

        model_key = write_json(
            model_artifact,
            f"{MODEL_PREFIX}{model_id}_anomaly_model.json"
        )

        experiment = {
            "experiment_id": str(uuid.uuid4()),
            "model_id": model_id,
            "experiment_name": "settlenet-anomaly-training",
            "started_at_utc": trained_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETED",
            "training_data": f"s3://{bucket}/{source_key}",
            "validation_report": f"s3://{bucket}/{report_key}",
            "metrics": {
                "rows_used": len(amounts),
                "median_amount": round(median_amount, 4),
                "median_absolute_deviation": round(mad, 4),
                "anomalies_detected": len(anomalies),
                "anomaly_rate": round(len(anomalies) / len(amounts), 4)
            },
            "anomalies": anomalies,
            "model_artifact": f"s3://{bucket}/{model_key}"
        }

        experiment_key = write_json(
            experiment,
            f"{EXPERIMENT_PREFIX}{model_id}_experiment.json"
        )

        return {
            "training_status": "COMPLETED",
            "model_id": model_id,
            "model_artifact": f"s3://{bucket}/{model_key}",
            "experiment_report": f"s3://{bucket}/{experiment_key}",
            "anomalies_detected": len(anomalies)
        }

    except Exception as error:
        return {
            "training_status": "ERROR",
            "error": str(error)
        }


def write_json(payload, key):
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json"
    )
    return key
