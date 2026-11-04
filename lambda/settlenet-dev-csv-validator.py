import json
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3

s3 = boto3.client("s3")

BUCKET_NAME = os.environ.get("BUCKET_NAME", "settlenet-dev-apar-7")
INCOMING_PREFIX = "incoming/"
REPORT_PREFIX = "validation-reports/"

REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "amount",
    "currency",
    "transaction_date",
    "status",
]

ALLOWED_STATUSES = {"completed", "pending", "failed"}


def lambda_handler(event, context):
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"], encoding="utf-8")

        if bucket != BUCKET_NAME or not key.startswith(INCOMING_PREFIX):
            results.append({
                "object_key": key,
                "skipped": True,
                "reason": "Object is outside the incoming/ prefix"
            })
            continue

        result = validate_csv(bucket, key)
        results.append(result)

    return {
        "statusCode": 200,
        "validation_results": results
    }


def validate_csv(bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8-sig")
        rows = parse_csv(content)

        errors = []
        warnings = []

        if not rows:
            errors.append("CSV file contains no data rows")

        for row_number, row in enumerate(rows, start=2):
            missing = [
                column
                for column in REQUIRED_COLUMNS
                if column not in row or not str(row[column]).strip()
            ]

            if missing:
                errors.append(
                    f"Row {row_number}: missing or empty fields: {', '.join(missing)}"
                )
                continue

            try:
                amount = float(row["amount"])
                if amount <= 0:
                    errors.append(
                        f"Row {row_number}: amount must be greater than zero"
                    )
            except ValueError:
                errors.append(f"Row {row_number}: amount is not numeric")

            status = str(row["status"]).strip().lower()
            if status not in ALLOWED_STATUSES:
                errors.append(
                    f"Row {row_number}: invalid status '{row['status']}'. "
                    f"Allowed values: {', '.join(sorted(ALLOWED_STATUSES))}"
                )

            date_value = str(row["transaction_date"]).strip().strip('"').strip("'")
            date_formats = [
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d-%m-%Y",
                "%Y/%m/%d"
            ]

            parsed_date = None
            for date_format in date_formats:
                try:
                    parsed_date = datetime.strptime(date_value, date_format)
                    break
                except ValueError:
                    continue

            if parsed_date is None:
                errors.append(
                    f"Row {row_number}: transaction_date '{date_value}' "
                    "is not a recognised date format"
                )

        report = {
            "validation_id": str(uuid.uuid4()),
            "source_object": f"s3://{bucket}/{key}",
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows_checked": len(rows),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "status": "PASSED" if not errors else "FAILED",
            "errors": errors,
            "warnings": warnings,
        }

        report_key = write_report(report, key)
        report["report_location"] = f"s3://{bucket}/{report_key}"

        return report

    except Exception as error:
        return {
            "source_object": f"s3://{bucket}/{key}",
            "status": "ERROR",
            "error": str(error)
        }


def parse_csv(content):
    import csv
    import io

    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def write_report(report, source_key):
    file_name = os.path.basename(source_key)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_key = f"{REPORT_PREFIX}{timestamp}_{file_name}_validation.json"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=report_key,
        Body=json.dumps(report, indent=2).encode("utf-8"),
        ContentType="application/json"
    )

    return report_key
