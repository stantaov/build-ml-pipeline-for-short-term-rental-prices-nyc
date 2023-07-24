#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def add_argument(parser, arg_name, arg_type, arg_help):
    parser.add_argument(
        arg_name,
        type=arg_type,
        help=arg_help,
        required=True
    )

def go(args):
    """
    This function performs basic cleaning on the data and logs the results in Weights & Biases
    """
    with wandb.init(job_type="basic_cleaning") as run:
        run.config.update(args)

        # Download input artifact. This will also log that this script is using this
        # particular version of the artifact
        artifact_local_path = run.use_artifact(args.input_artifact).file()
        logger.info(f"Downloading artifact to {artifact_local_path}")
        
        # Load data to a DataFrame
        logger.info("Loading data")
        df = pd.read_csv(artifact_local_path)
        print(df.head())

        # Drop outliers
        logger.info("Dropping outliers in the price column")
        df = df[df['price'].between(args.min_price, args.max_price)]

        # Restrict long and lat
        idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
        df = df[idx].copy()


        # Convert last_review to datetime
        df['last_review'] = pd.to_datetime(df['last_review'])

        # Save the results to a CSV file
        logger.info("Save the results to CSV")
        df.to_csv(args.output_artifact, index=False)

        # Build artifact
        artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )

        logger.info("Log artifact: clean_sample.csv")
        artifact.add_file(args.output_artifact)
        run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    add_argument(parser, "--input_artifact", str, "Name of the artifact to do preprocessing on")
    add_argument(parser, "--output_artifact", str, "Name of the clean artifact")
    add_argument(parser, "--output_type", str, "clean_sample")
    add_argument(parser, "--output_description", str, "Description of the preprocessed data")
    add_argument(parser, "--min_price", float, "Min price considered for the prediction column")
    add_argument(parser, "--max_price", float, "Max price considered for the prediction column")

    args = parser.parse_args()

    go(args)
