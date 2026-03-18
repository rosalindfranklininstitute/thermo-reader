# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import sys
import os
import argparse

from . import thermo

from pathlib import Path

from ms_nexus_tools.api import args as nxargs

from .load_thermo import (
    RawFileReaderAdapter,
    Device,
    Enum,
    DataUnits,
    ListTrailerExtraFields,
    IScanFilter,
    GetChromatogram,
    ReadScanInformation,
    GetSpectrum,
    GetAverageSpectrum,
    ReadAllSpectra,
    CalculateMassPrecision,
    SampleType,
    is_os_windows,
)


def test():
    parser = argparse.ArgumentParser(prog="thermo")

    nxargs.add_arguments(parser, thermo.ProcessArgs)

    args, config_dict = thermo.ProcessArgs.parse_args(parser)

    process_args = thermo.ProcessArgs(**vars(args))

    thermo.process(process_args, config_dict)


def main():

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print("No RAW file specified!")
        exit()

    # Check to see if the specified RAW file exists
    if not os.path.isfile(filename):
        print("The file doesn't exist in the specified location - {}".format(filename))
        exit()

    # Create the IRawDataPlus object for accessing the RAW file
    rawFile = RawFileReaderAdapter.FileFactory(filename)

    if not rawFile.IsOpen or rawFile.IsError:
        print("Unable to access the RAW file using the RawFileReader class!")
        exit()

    # Check for any errors in the RAW file
    if rawFile.IsError:
        print("Error opening ({}) - {}".format(rawFile.FileError, filename))
        exit()

    # Check if the RAW file is being acquired
    if rawFile.InAcquisition:
        print("RAW file still being acquired - {}".format(filename))
        exit()

    # Get the number of instruments (controllers) present in the RAW file
    # and set the selected instrument to the MS instrument, first instance
    # of it
    print("The RAW file has data from {} instruments".format(rawFile.InstrumentCount))

    rawFile.SelectInstrument(Device.MS, 1)

    # Get the first and last scan from the RAW file
    firstScanNumber = rawFile.RunHeaderEx.FirstSpectrum
    lastScanNumber = rawFile.RunHeaderEx.LastSpectrum

    # Get the start and end time from the RAW file
    startTime = rawFile.RunHeaderEx.StartTime
    endTime = rawFile.RunHeaderEx.EndTime

    # Get some information from the header portions of the RAW file and
    # display that information.  The information is general information
    # pertaining to the RAW file.
    print("General File Information:")
    print("   RAW file: {}".format(rawFile.FileName))
    print("   RAW file version: {}".format(rawFile.FileHeader.Revision))
    print("   Creation date: {}".format(rawFile.FileHeader.CreationDate))
    print("   Operator: {}".format(rawFile.FileHeader.WhoCreatedId))
    print("   Number of instruments: {}".format(rawFile.InstrumentCount))
    print("   Description: {}".format(rawFile.FileHeader.FileDescription))
    print("   Instrument model: {}".format(rawFile.GetInstrumentData().Model))
    print("   Instrument name: {}".format(rawFile.GetInstrumentData().Name))
    print("   Serial number: {}".format(rawFile.GetInstrumentData().SerialNumber))
    print("   Software version: {}".format(rawFile.GetInstrumentData().SoftwareVersion))
    print("   Firmware version: {}".format(rawFile.GetInstrumentData().HardwareVersion))
    print(
        "   Units: {}".format(
            Enum.GetName(DataUnits, rawFile.GetInstrumentData().Units)
        )
    )
    print("   Mass resolution: {:.3f}".format(rawFile.RunHeaderEx.MassResolution))
    print("   Number of scans: {}".format(rawFile.RunHeaderEx.SpectraCount))
    print("   Scan range: {} - {}".format(firstScanNumber, lastScanNumber))
    print("   Time range: {:.2f} - {:.2f}".format(startTime, endTime))
    print(
        "   Mass range: {:.4f} - {:.4f}".format(
            rawFile.RunHeaderEx.LowMass, rawFile.RunHeaderEx.HighMass
        )
    )
    print()

    # Get information related to the sample that was processed
    print("Sample Information:")
    print("   Sample name: {}".format(rawFile.SampleInformation.SampleName))
    print("   Sample id: {}".format(rawFile.SampleInformation.SampleId))
    print(
        "   Sample type: {}".format(
            Enum.GetName(SampleType, rawFile.SampleInformation.SampleType)
        )
    )
    print("   Sample comment: {}".format(rawFile.SampleInformation.Comment))
    print("   Sample vial: {}".format(rawFile.SampleInformation.Vial))
    print("   Sample volume: {}".format(rawFile.SampleInformation.SampleVolume))
    print(
        "   Sample injection volume: {}".format(
            rawFile.SampleInformation.InjectionVolume
        )
    )
    print("   Sample row number: {}".format(rawFile.SampleInformation.RowNumber))
    print(
        "   Sample dilution factor: {}".format(rawFile.SampleInformation.DilutionFactor)
    )
    print()

    # Read the first instrument method (most likely for the MS portion of
    # the instrument).  NOTE: This method reads the instrument methods
    # from the RAW file but the underlying code uses some Microsoft code
    # that hasn't been ported to Linux or MacOS.  Therefore this method
    # won't work on those platforms therefore the check for Windows.
    if is_os_windows():
        deviceNames = rawFile.GetAllInstrumentNamesFromInstrumentMethod()
        for device in deviceNames:
            print("Instrument method: {}".format(device))
        print()

    # Display all of the trailer extra data fields present in the RAW file
    ListTrailerExtraFields(rawFile)

    # Get the number of filters present in the RAW file
    numberFilters = rawFile.GetFilters().Count

    # Get the scan filter for the first and last spectrum in the RAW file
    firstFilter = IScanFilter(rawFile.GetFilterForScanNumber(firstScanNumber))
    lastFilter = IScanFilter(rawFile.GetFilterForScanNumber(lastScanNumber))

    print("Filter Information:")
    print("   Scan filter (first scan): {}".format(firstFilter.ToString()))
    print("   Scan filter (last scan): {}".format(lastFilter.ToString()))
    print("   Total number of filters: {}".format(numberFilters))
    print()

    # Get the BasePeak chromatogram for the MS data
    GetChromatogram(rawFile, firstScanNumber, lastScanNumber, True)

    # Read the scan information for each scan in the RAW file
    ReadScanInformation(rawFile, firstScanNumber, lastScanNumber, True)

    # Get a spectrum from the RAW file.
    GetSpectrum(rawFile, firstScanNumber, firstFilter.ToString(), True)

    # Get a average spectrum from the RAW file for the first 15 scans in the file.
    GetAverageSpectrum(rawFile, 1, 15, False)

    # Read each spectrum
    ReadAllSpectra(rawFile, firstScanNumber, lastScanNumber, True)

    # Calculate the mass precision for a spectrum
    CalculateMassPrecision(rawFile, 1)

    # Close (dispose) the RAW file
    print()
    print("Closing {}".format(filename))

    rawFile.Dispose()
