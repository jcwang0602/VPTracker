# Copyright (c) ModelScope Contributors. All rights reserved.
# Packaging adapted for the VPTracker SFT and inference source subset.
from pathlib import Path
import os
import re

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements(path):
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            requirements.extend(read_requirements(path.parent / line[3:].strip()))
        else:
            requirements.append(line)
    return requirements


if __name__ == "__main__":
    os.chdir(ROOT)
    version_text = (ROOT / "swift/version.py").read_text(encoding="utf-8")
    version = re.search(r"^__version__\s*=\s*['\"](.+?)['\"]", version_text, re.M).group(1)
    setup(
        name="ms_swift",
        version=version,
        description="ms-swift SFT and inference source subset maintained for VPTracker",
        long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        author="DAMO ModelScope teams; VPTracker maintainers",
        url="https://github.com/jcwang0602/VPTracker",
        project_urls={"Upstream": "https://github.com/modelscope/ms-swift"},
        packages=find_packages(include=("swift", "swift.*")),
        include_package_data=True,
        package_data={"swift": ["config/*.json", "dataset/data/*.json", "loss_scale/config/*.json"]},
        python_requires=">=3.10",
        license="Apache-2.0",
        license_files=("LICENSE",),
        install_requires=read_requirements(ROOT / "requirements.txt"),
        entry_points={"console_scripts": ["swift=swift.cli.main:cli_main"]},
        zip_safe=False,
    )
