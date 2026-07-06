import numpy as np

from .load_thermo import (
    Device,
    IRawDataExtended,
    ScanStatistics,
)


class MSInstrumentData:
    def __init__(self, rawFile: IRawDataExtended, index):
        self.raw_file = rawFile
        self.raw_file.SelectInstrument(Device.MS, index)
        self._data = self.raw_file.GetInstrumentData()

        self.first_scan_number = self.raw_file.RunHeaderEx.FirstSpectrum
        self.last_scan_number = self.raw_file.RunHeaderEx.LastSpectrum

        self.first_scan_statistics = rawFile.GetScanStatsForScanNumber(
            self.first_scan_number
        )

    def stored_mass_resolution(self) -> float:
        return self.raw_file.RunHeaderEx.MassResolution

    def is_centroid_scan(self):
        return self.first_scan_statistics.IsCentroidScan

    def scan_range(self) -> range:
        return range(self.first_scan_number, self.last_scan_number)

    def get_centroid_stream(
        self, scan_number: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centroidStream = self.raw_file.GetCentroidStream(scan_number, False)

        masses = np.array([m for m in centroidStream.Masses])
        intensities = np.array([m for m in centroidStream.Intensities])
        charges = np.array([m for m in centroidStream.Charges])

        return masses, intensities, charges

    def get_segmented_scan(self, scan_number: int) -> tuple[np.ndarray, np.ndarray]:
        scan_statistics = ScanStatistics()
        segmentedScan = self.raw_file.GetSegmentedScanFromScanNumber(
            scan_number, scan_statistics
        )
        masses = np.array([m for m in segmentedScan.Positions])
        intensities = np.array([m for m in segmentedScan.Intensities])

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
        segmentedScan = self.raw_file.GetSegmentedScanFromScanNumber(
            scan_number, scan_statistics
        )

        if segmentedScan.Positions is None:
            raise IndexError(f"{scan_number} not a valid scan number.")
        spec, _ = np.histogram(
            segmentedScan.Positions, bins=mass_axis, weights=segmentedScan.Intensities
        )
        return spec, len(segmentedScan.Intensities)
