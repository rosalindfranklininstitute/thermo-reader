# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from .load_thermo import (
    Device,
    IRawDataExtended,
    ScanStatistics,
    to_py_datetime,
)


class MSInstrumentData:
    def __init__(self, raw_file: IRawDataExtended, index):
        self.raw_file = raw_file
        self.raw_file.SelectInstrument(Device.MS, index)
        self._data = self.raw_file.GetInstrumentData()

        self.first_scan_number = self.raw_file.RunHeaderEx.FirstSpectrum
        self.last_scan_number = self.raw_file.RunHeaderEx.LastSpectrum
        self.creatoin_time = to_py_datetime(self.raw_file.CreationDate)

        self.first_scan_statistics = self.raw_file.GetScanStatsForScanNumber(
            self.first_scan_number
        )

    def stored_mass_resolution(self) -> float:
        return self.raw_file.RunHeaderEx.MassResolution

    def is_centroid_scan(self) -> bool:
        return self.first_scan_statistics.IsCentroidScan

    def scan_range(self) -> range:
        return range(self.first_scan_number, self.last_scan_number)

    def get_centroid_stream(
        self, scan_number: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centroid_stream = self.raw_file.GetCentroidStream(scan_number, False)

        masses = np.array(list(centroid_stream.Masses))
        intensities = np.array(list(centroid_stream.Intensities))
        charges = np.array(list(centroid_stream.Charges))

        return masses, intensities, charges

    def get_segmented_scan(self, scan_number: int) -> tuple[np.ndarray, np.ndarray]:
        scan_statistics = ScanStatistics()
        segmented_scan = self.raw_file.GetSegmentedScanFromScanNumber(
            scan_number, scan_statistics
        )
        masses = np.array(list(segmented_scan.Positions))
        intensities = np.array(list(segmented_scan.Intensities))

        return masses, intensities

    def get_scan_number(self, scan_index) -> int:
        return int(self.first_scan_number + scan_index)

    def get_mass_range(self) -> tuple[float, float]:
        stats = self.raw_file.GetScanStatsForScanNumber(self.first_scan_number)
        return stats.LowMass, stats.HighMass

    def get_scan_times(self) -> np.ndarray:
        return np.array(
            [
                self.raw_file.GetScanStatsForScanNumber(ii).StartTime * 60
                for ii in self.scan_range()
            ]
        )

    def get_spectra_on_mass(
        self, scan_number: int, mass_axis: np.ndarray
    ) -> tuple[np.ndarray, int]:
        scan_statistics = ScanStatistics()
        segmented_scan = self.raw_file.GetSegmentedScanFromScanNumber(
            scan_number, scan_statistics
        )

        if segmented_scan.Positions is None:
            raise IndexError(f"{scan_number} not a valid scan number.")
        spec, _ = np.histogram(
            segmented_scan.Positions, bins=mass_axis, weights=segmented_scan.Intensities
        )
        return spec, len(segmented_scan.Intensities)
