"""Register source ancestry before a Batch, EC2 or initial-seed publication."""

import argparse
import os

import boto3

from src.shared.artifact_publication import register_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--prefix", default=os.environ.get("FF_MODEL_S3_PREFIX", "models"))
    args = parser.parse_args()
    record = register_source(boto3.client("s3"), args.bucket, args.prefix, args.sha)
    print(f"Registered {record['source_sha']} at source order {record['source_order']}")


if __name__ == "__main__":
    main()
