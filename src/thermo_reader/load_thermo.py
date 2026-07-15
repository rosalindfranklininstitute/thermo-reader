# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

from pythonnet import load
from pathlib import Path

load("coreclr")
import clr

dll_directory = Path(__file__).parent

clr.AddReference(str(Path(dll_directory, "dlls", "ThermoFisher.CommonCore.Data")))
clr.AddReference(
    str(Path(dll_directory, "dlls", "ThermoFisher.CommonCore.RawFileReader"))
)
clr.AddReference(
    str(Path(dll_directory, "dlls", "ThermoFisher.CommonCore.BackgroundSubtraction"))
)
clr.AddReference(
    str(Path(dll_directory, "dlls", "ThermoFisher.CommonCore.MassPrecisionEstimator"))
)

import datetime as dt

from System import Enum, Environment, DateTime
from System.Collections.Generic import List

from ThermoFisher.CommonCore.Data import ToleranceUnits
from ThermoFisher.CommonCore.Data import Extensions
from ThermoFisher.CommonCore.Data.Business import (
    ChromatogramSignal as ChromatogramSignal,
    ChromatogramTraceSettings as ChromatogramTraceSettings,
    DataUnits as DataUnits,
    Device as Device,
    GenericDataTypes as GenericDataTypes,
    SampleType as SampleType,
    Scan as Scan,
    ScanStatistics as ScanStatistics,
    TraceType as TraceType,
)
from ThermoFisher.CommonCore.Data.FilterEnums import IonizationModeType, MSOrderType
from ThermoFisher.CommonCore.Data.Interfaces import (
    IChromatogramSettings as IChromatogramSettings,
    IRawDataExtended as IRawDataExtended,
    IScanEventBase as IScanEventBase,
    IScanFilter as IScanFilter,
    RawFileClassification as RawFileClassification,
)
from ThermoFisher.CommonCore.MassPrecisionEstimator import (
    PrecisionEstimate as PrecisionEstimate,
)
from ThermoFisher.CommonCore.RawFileReader import (
    RawFileReaderAdapter as RawFileReaderAdapter,
)


def is_os_windows() -> bool:
    return "Windows" in str(Environment.OSVersion)


def to_py_datetime(cs_datetime: DateTime) -> dt.datetime:
    return dt.datetime(
        year=cs_datetime.Year,
        month=cs_datetime.Month,
        day=cs_datetime.Day,
        hour=cs_datetime.Hour,
        minute=cs_datetime.Minute,
        second=cs_datetime.Second,
        microsecond=cs_datetime.Millisecond * 1000 + cs_datetime.Microsecond,
    )


def inspect(thing) -> None:
    print(f"{thing}")
    for ss in dir(thing):
        if len(ss) > 0 and ss[0].isupper() and "_" not in ss:
            try:
                value = getattr(thing, ss)
            except ValueError as e:
                print(f"{ss}: {e}")
            else:
                print(f"{ss}: {value}")


def ListTrailerExtraFields(raw_file) -> None:
    """
    Reads and reports the trailer extra data fields present in the RAW
    file.

    Args:
        raw_file (IRawDataPlus): the RAW file.
    """
    # Get the Trailer Extra data fields present in the RAW file
    trailer_fields = raw_file.GetTrailerExtraHeaderInformation()

    # Display each value
    i = 0
    print("Trailer Extra Data Information:")

    for i, field in enumerate(trailer_fields):
        print(
            "   Field {} = {} storing data of type {}".format(
                i, field.Label, Enum.GetName(GenericDataTypes, field.DataType)
            )
        )

    print()


def GetChromatogram(raw_file, start_scan, end_scan, output_data) -> None:
    """
    Reads the base peak chromatogram for the RAW file.

    Args:
        raw_file (IRawDataPlus): the RAW file being read.
        start_scan (int): start scan for the chromatogram.
        end_scan (int): end scan for the chromatogram.
        output_data (bool): the output data flag.
    """
    # Define the settings for getting the Base Peak chromatogram
    settings = ChromatogramTraceSettings(TraceType.BasePeak)

    # Get the chromatogram from the RAW file.
    data = raw_file.GetChromatogramData([settings], start_scan, end_scan)

    # Split the data into the chromatograms
    trace = ChromatogramSignal.FromChromatogramData(data)

    if output_data:
        print(f"Number of traces: {trace.Length}")
        for tr_num, tr in enumerate(trace):
            print(f"Base Peak chromatogram ({tr.Length} points)")
            for ii in range(tr.Length):
                print(
                    f"  Trace[{tr_num}]: {ii} - Time {tr.Times[ii]:.3f}, Int {tr.Intensities[ii]:.0f}"
                )

    print()


def ReadScanInformation(
    raw_file, first_scan_number, last_scan_number, output_data
) -> None:
    """
    Reads the general scan information for each scan in the RAW file
    using the scan filter object and also the trailer extra data
    section for that same scan.

    Args:
        raw_file (IRawDataPlus): the RAW file being read.
        first_scan_number (int): the first scan in the RAW file.
        last_scan_number (int): the last scan in the RAW file.
        output_data (bool): the output data flag.
    """
    # Read each scan in the RAW File
    for scan in range(first_scan_number, last_scan_number):
        # Get the retention time for this scan number.  This is one of
        # two comparable functions that will convert between retention
        # time and scan number.
        time = raw_file.RetentionTimeFromScanNumber(scan)

        # Get the scan filter for this scan number
        scan_filter = IScanFilter(raw_file.GetFilterForScanNumber(scan))

        # Get the scan event for this scan number
        scan_event = IScanEventBase(raw_file.GetScanEventForScanNumber(scan))

        # Get the ionization_mode, MS2 precursor mass, collision
        # energy, and isolation width for each scan
        if scan_filter.MSOrder == MSOrderType.Ms2:
            # Get the reaction information for the first precursor
            reaction = scan_event.GetReaction(0)

            precursor_mass = reaction.PrecursorMass
            collision_energy = reaction.CollisionEnergy
            isolation_width = reaction.IsolationWidth
            monoisotopic_mass = 0.0
            master_scan = 0
            ionization_mode = scan_filter.IonizationMode
            order = scan_filter.MSOrder

            # Get the trailer extra data for this scan and then look
            # for the monoisotopic m/z value in the trailer extra data
            # list
            trailer_data = raw_file.GetTrailerExtraInformation(scan)

            for i in range(trailer_data.Length):
                if trailer_data.Labels[i] == "Monoisotopic M/Z:":
                    monoisotopic_mass = float(trailer_data.Values[i])
                elif trailer_data.Labels[i] in (
                    "Master Scan Number:",
                    "Master Scan Number",
                    "Master Index:",
                ):
                    master_scan = int(trailer_data.Values[i])

            if output_data:
                print(
                    """Scan number {} @ time {:.2f} - Master scan = {}, Ionization mode={},\
                                MS Order={}, Precursor mass={:.4f}, Monoisotopic Mass = {:.4f},\
                                Collision energy={:.2f}, Isolation width={:.2f}""".format(
                        scan,
                        time,
                        master_scan,
                        Enum.GetName(IonizationModeType, ionization_mode),
                        Enum.GetName(MSOrderType, order),
                        precursor_mass,
                        monoisotopic_mass,
                        collision_energy,
                        isolation_width,
                    )
                )

        elif scan_filter.MSOrder == MSOrderType.Ms:
            scan_dependents = raw_file.GetScanDependents(scan, 5)

            print(
                "Scan number {} @ time {:.2f} - Instrument type={}, Number dependent scans={}".format(
                    scan,
                    time,
                    Enum.GetName(
                        RawFileClassification, scan_dependents.RawFileInstrumentType
                    ),
                    scan_dependents.ScanDependentDetailArray.Length,
                )
            )


def GetSpectrum(raw_file, scan_number, scan_filter, output_data) -> None:
    """
    Gets the spectrum from the RAW file.

    Args:
        raw_file (IRawDataPlus): the RAW file being read.
        scan_number (int): the scan number being read.
        scan_filter (str): the scan filter for that scan.
        output_data (bool): the output data flag.
    """
    # Check for a valid scan filter
    if not scan_filter:
        return

    # Get the scan statistics from the RAW file for this scan number
    scan_statistics = raw_file.GetScanStatsForScanNumber(scan_number)

    # Check to see if the scan has centroid data or profile data.  Depending upon the
    # type of data, different methods will be used to read the data.  While the ReadAllSpectra
    # method demonstrates reading the data using the Scan.FromFile method, generating the
    # Scan object takes more time and memory to do, so that method isn't optimum.
    if scan_statistics.IsCentroidScan:
        # Get the centroid (label) data from the RAW file for this
        # scan
        centroid_stream = raw_file.GetCentroidStream(scan_number, False)

        print(
            "Spectrum (centroid/label) {} - {} points".format(
                scan_number, centroid_stream.Length
            )
        )

        # Print the spectral data (mass, intensity, charge values).
        # Not all of the information in the high resolution centroid
        # (label data) object is reported in this example.  Please
        # check the documentation for more information about what is
        # available in high resolution centroid (label) data.
        if output_data:
            for i in range(centroid_stream.Length):
                print(
                    "  {} - {:.4f}, {:.0f}, {:.0f}".format(
                        i,
                        centroid_stream.Masses[i],
                        centroid_stream.Intensities[i],
                        centroid_stream.Charges[i],
                    )
                )
            print()

    else:
        # Get the segmented (low res and profile) scan data
        segmented_scan = raw_file.GetSegmentedScanFromScanNumber(
            scan_number, scan_statistics
        )

        print(
            "Spectrum (normal data) {} - {} points".format(
                scan_number, segmented_scan.Positions.Length
            )
        )

        # Print the spectral data (mass, intensity values)
        if output_data:
            for i in range(segmented_scan.Positions.Length):
                print(
                    "  {} - {:.4f}, {:.0f}".format(
                        i, segmented_scan.Positions[i], segmented_scan.Intensities[i]
                    )
                )
            print()


def GetAverageSpectrum(
    raw_file, first_scan_number, last_scan_number, output_data
) -> None:
    """
    Gets the average spectrum from the RAW file.

    Args:
        raw_file (IRawDataPlus): the RAW file being read.
        first_scan_number (int): the first scan to consider for the averaged spectrum.
        last_scan_number (int): the last scan to consider for the averaged spectrum.
        output_data (bool): the output data flag.
    """
    # Create the mass options object that will be used when averaging
    # the scans
    options = Extensions.DefaultMassOptions(raw_file)

    options.ToleranceUnits = ToleranceUnits.ppm
    options.Tolerance = 5.0

    # Get the scan filter for the first scan.  This scan filter will be used to located
    # scans within the given scan range of the same type
    scan_filter = IScanFilter(raw_file.GetFilterForScanNumber(first_scan_number))

    # Get the average mass spectrum for the provided scan range. In addition to getting the
    # average scan using a scan range, the library also provides a similar method that takes
    # a time range.
    average_scan = Extensions.AverageScansInScanRange(
        raw_file, first_scan_number, last_scan_number, scan_filter, options
    )

    if average_scan.HasCentroidStream:
        print("Average spectrum ({} points)".format(average_scan.CentroidScan.Length))

        # Print the spectral data (mass, intensity values)
        if output_data:
            for i in range(average_scan.CentroidScan.Length):
                print(
                    "  {:.4f} {:.0f}".format(
                        average_scan.CentroidScan.Masses[i],
                        average_scan.CentroidScan.Intensities[i],
                    )
                )

    # This example uses a different method to get the same average spectrum that was calculated in the
    # previous portion of this method.  Instead of passing the start and end scan, a list of scans will
    # be passed to the GetAveragedMassSpectrum function.
    scans = List[int]()
    for v in [1, 6, 7, 9, 11, 12, 14]:
        scans.Add(v)

    average_scan = Extensions.AverageScans(raw_file, scans, options)

    if average_scan.HasCentroidStream:
        print("Average spectrum ({} points)", average_scan.CentroidScan.Length)

        # Print the spectral data (mass, intensity values)
        if output_data:
            for i in range(average_scan.CentroidScan.Length):
                print(
                    "  {:.4f} {:.0f}".format(
                        average_scan.CentroidScan.Masses[i],
                        average_scan.CentroidScan.Intensities[i],
                    )
                )

    print()


def ReadAllSpectra(raw_file, first_scan_number, last_scan_number, output_data) -> None:
    """
    Read all spectra in the RAW file.

    Args:
        raw_file (IRawDataPlus): the raw file.
        first_scan_number (int): the first scan number.
        last_scan_number (int): the last scan number.
        output_data (bool): the output data flag.
    """
    for scan_number in range(first_scan_number, last_scan_number):
        try:
            # Get the scan filter for the spectrum
            scan_filter = IScanFilter(
                raw_file.GetFilterForScanNumber(first_scan_number)
            )

            if not scan_filter.ToString():
                continue

            # Get the scan from the RAW file.  This method uses the Scan.FromFile method which returns a
            # Scan object that contains both the segmented and centroid (label) data from an FTMS scan
            # or just the segmented data in non-FTMS scans.  The GetSpectrum method demonstrates an
            # alternative method for reading scans.
            scan = Scan.FromFile(raw_file, scan_number)

            # If that scan contains FTMS data then Centroid stream
            # will be populated so check to see if it is present.
            label_size = scan.CentroidScan.Length if scan.HasCentroidStream else 0

            # For non-FTMS data, the preferred data will be populated
            if scan.PreferredMasses is not None:
                data_size = scan.PreferredMasses.Length
            else:
                data_size = 0

            if output_data:
                print(
                    "Spectrum {} - {}: normal {}, label {} points".format(
                        scan_number, scan_filter.ToString(), data_size, label_size
                    )
                )

        except ValueError as ex:
            print("Error reading spectrum {} - {}".format(scan_number, str(ex)))


def CalculateMassPrecision(raw_file, scan_number) -> None:
    """
    Calculates the mass precision for a spectrum.

    Args:
        raw_file (IRawDataPlus): the RAW file being read.
        scan_number (int): the scan to process.
    """
    # Get the scan from the RAW file
    scan = Scan.FromFile(raw_file, scan_number)

    # Get the scan event and from the scan event get the analyzer type for this scan
    scan_event = IScanEventBase(raw_file.GetScanEventForScanNumber(scan_number))

    # Get the trailer extra data to get the ion time for this file
    log_entry = raw_file.GetTrailerExtraInformation(scan_number)

    trailer_headings = List[str]()
    trailer_values = List[str]()
    for i in range(log_entry.Length):
        trailer_headings.Add(log_entry.Labels[i])
        trailer_values.Add(log_entry.Values[i])

    # Create the mass precision estimate object
    precision_estimate = PrecisionEstimate()

    # Get the ion time from the trailer extra data values
    ion_time = precision_estimate.GetIonTime(
        scan_event.MassAnalyzer, scan, trailer_headings, trailer_values
    )

    # Calculate the mass precision for the scan
    list_results = precision_estimate.GetMassPrecisionEstimate(
        scan, scan_event.MassAnalyzer, ion_time, raw_file.RunHeader.MassResolution
    )

    # Output the mass precision results
    if list_results.Count:
        print("Mass Precision Results:")

        for result in list_results:
            print(
                "Mass {:.5f}, mmu = {:.3f}, ppm = {:.2f}".format(
                    result.Mass, result.MassAccuracyInMmu, result.MassAccuracyInPpm
                )
            )
