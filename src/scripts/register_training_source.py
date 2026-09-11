"""Register source ancestry before Batch, EC2 or initial-seed publication."""

import argparse
import os

from src.artifacts.source import register_source


def main():
    import boto3

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--prefix", default=os.environ.get("FF_MODEL_S3_PREFIX", "models"))
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    record = register_source(boto3.client("s3"), args.bucket, args.prefix, args.sha, args.repo)
    print(f"Registered {record['source_sha']} at source order {record['source_order']}")


if __name__ == "__main__":
    main()
