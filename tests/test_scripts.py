import subprocess


def test_common_shell_does_not_invent_a_default_aws_profile() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'unset AWS_PROFILE; source scripts/common.sh; [[ -z "${AWS_PROFILE+x}" ]]',
        ],
        check=False,
    )

    assert result.returncode == 0


def test_aws_cli_adds_profile_only_when_explicitly_configured() -> None:
    command = (
        "source scripts/common.sh; "
        "aws() { printf '%s\\n' \"$*\"; }; "
        "aws_cli sts get-caller-identity"
    )

    default_chain = subprocess.run(
        ["bash", "-c", f"unset AWS_PROFILE; {command}"],
        check=True,
        capture_output=True,
        text=True,
    )
    named_profile = subprocess.run(
        ["bash", "-c", f"export AWS_PROFILE=snail-data; {command}"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--profile" not in default_chain.stdout
    assert "--profile snail-data" in named_profile.stdout
