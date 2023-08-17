# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import bz2
import gzip
import hashlib
import os
import pathlib
import ssl
import tarfile
import urllib
import urllib.error
import urllib.request
import zipfile

from tqdm import tqdm

FILE_TYPE_ALIASES = {".tbz": (".tar", ".bz2"), ".tbz2": (".tar", ".bz2"), ".tgz": (".tar", ".gz")}

ARCHIVE_TYPE_SUFFIX = [".tar", ".zip"]

COMPRESS_TYPE_SUFFIX = [".bz2", ".gz"]


def detect_file_type(filename):  # pylint: disable=inconsistent-return-statements
    """Detect file type by suffixes and return tuple(suffix, archive_type, compression)."""
    suffixes = pathlib.Path(filename).suffixes
    if not suffixes:
        raise RuntimeError(f"File `{filename}` has no suffixes that could be used to detect.")
    suffix = suffixes[-1]

    # Check if the suffix is a known alias.
    if suffix in FILE_TYPE_ALIASES:
        return suffix, FILE_TYPE_ALIASES[suffix][0], FILE_TYPE_ALIASES[suffix][1]

    # Check if the suffix is an archive type.
    if suffix in ARCHIVE_TYPE_SUFFIX:
        return suffix, suffix, None

    # Check if the suffix is a compression.
    if suffix in COMPRESS_TYPE_SUFFIX:
        # Check for suffix hierarchy.
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]
            # Check if the suffix2 is an archive type.
            if suffix2 in ARCHIVE_TYPE_SUFFIX:
                return suffix2 + suffix, suffix2, suffix
        return suffix, None, suffix


class DownLoad(object):
    """Base utility class for downloading."""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/92.0.4515.131 Safari/537.36"
    )

    @staticmethod
    def calculate_md5(file_path, chunk_size=1024 * 1024):
        md5 = hashlib.md5()
        with open(file_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def check_md5(self, file_path, md5=None):
        return md5 == self.calculate_md5(file_path)

    @staticmethod
    def extract_tar(from_path, to_path=None, compression=None):
        """Extract tar format file."""

        with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
            tar.extractall(to_path)

    @staticmethod
    def extract_zip(from_path, to_path=None, compression=None):
        """Extract zip format file."""

        compression_mode = zipfile.ZIP_BZIP2 if compression else zipfile.ZIP_STORED
        with zipfile.ZipFile(from_path, "r", compression=compression_mode) as zip_file:
            zip_file.extractall(to_path)

    def decompress(self, from_path, to_path=None):
        compress_file_open = {".bz2": bz2.open, ".gz": gzip.open}

        archive_extractors = {
            ".tar": self.extract_tar,
            ".zip": self.extract_zip,
        }

        suffix, archive_type, compression = detect_file_type(from_path)  # pylint: disable=unused-variable

        if not to_path:
            to_path = os.path.dirname(from_path)

        if not archive_type:
            compress = compress_file_open[compression]
            to_path = from_path.replace(suffix, "")
            with compress(from_path, "rb") as rf, open(to_path, "wb") as wf:
                wf.write(rf.read())
            return to_path

        decompress = archive_extractors[archive_type]
        decompress(from_path, to_path, compression)

        return to_path

    def download_file(self, url, file_path, chunk_size=1024):
        # Define request headers.
        headers = {"User-Agent": self.USER_AGENT}

        with open(file_path, "wb") as f:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request) as response:
                with tqdm(total=response.length, unit="B") as pbar:
                    for chunk in iter(lambda: response.read(chunk_size), b""):
                        if not chunk:
                            break
                        pbar.update(chunk_size)
                        f.write(chunk)

    def download_url(self, url, path="./", filename=None, md5=None):
        if not filename:
            filename = os.path.basename(url)

        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, filename)

        # Check if the file is exists.
        if os.path.isfile(file_path):
            if not md5 or self.check_md5(file_path, md5):
                return

        # Download the file.
        try:
            self.download_file(url, file_path)
        except (urllib.error.URLError, IOError) as e:
            if url.startswith("https"):
                url = url.replace("https", "http")
                try:
                    self.download_file(url, file_path)
                except (urllib.error.URLError, IOError):
                    # pylint: disable=protected-access
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.download_file(url, file_path)
                    ssl._create_default_https_context = ssl.create_default_context
            else:
                raise e

    def download_and_decompress(
        self, url, download_path, extract_path=None, filename=None, md5=None, remove_finished=False
    ):
        download_path = os.path.expanduser(download_path)

        if not filename:
            filename = os.path.basename(url)

        self.download_url(url, download_path, filename, md5)

        archive = os.path.join(download_path, filename)
        self.decompress(archive, extract_path)

        if remove_finished:
            os.remove(archive)
