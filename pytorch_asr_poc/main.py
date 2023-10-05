import os
import click

from asr import estimate_transcript


def _load_transcripts(data_dir: str) -> dict:
    """Loads ground truth files from `LJ Speech Dataset`.

    File identifiers and actual transcripts stored as pipe separated values.

    Args:
        data_dir: Location of the unzipped `LJ Speech Dataset`.
    Returns: Dictionary of speech ID and transcript.

    """
    d = {}
    with open(os.path.join(data_dir, "metadata.csv"), "r") as f:
        for line in f:
            d[line.split(sep="|")[0]] = line.split(sep="|")[1]
    return d


def _count_occurrence(strings: list[str], target: str) -> int:
    """Counts occurrence of target word in list of strings.

    Applies `lower` transform to both string and target before checking.

    Args:
        strings: List of strings
        target: Target substring
    Returns: Count of target
    """

    count = 0
    for string in strings:
        count += string.lower().count(target.lower())
    return count


@click.command()
@click.option(
    "--data-dir", help="Location of LJ Speech Dataset", type=click.STRING, required=True
)
@click.option(
    "--target",
    help="Target string to count occurrences of",
    type=click.STRING,
    required=True,
)
@click.option("--num-samples", help="Number of samples", type=click.INT)
def main(data_dir: str, target: str, num_samples: int | None = None) -> None:
    """
    Entry point function for counting occurrences of target string in
    LJ Speech Dataset.

    Args:
        data_dir: Location of the unzipped `LJ Speech Dataset`.
        target: Target substring
        num_samples: Number of speeches to process - always starts from beginning.

    """
    transcripts = _load_transcripts(data_dir=data_dir)
    if num_samples is None:
        num_samples = len(transcripts.keys())

    wavs = [
        os.path.join(data_dir, "wavs", f"{filename}.wav")
        for filename in transcripts.keys()
    ][0:num_samples]

    count = 0
    estimated_transcripts = []
    for wav in wavs:
        estimated_transcripts.append(estimate_transcript(filename=wav))
        count += 1
        print(f"Percentage complete: {100*count/len(wavs)}%")

    count_act = _count_occurrence(transcripts.values(), target)
    count_est = _count_occurrence(estimated_transcripts, target)

    print(f"Actual count of target word `{target}` is {count_act}.")
    print(f"Estimated count of target word `{target}` is {count_est}.")


if __name__ == "__main__":
    main()
