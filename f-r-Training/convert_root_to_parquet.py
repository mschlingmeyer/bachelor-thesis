import awkward as ak
import uproot as up
import os

from tqdm import tqdm
from optparse import OptionParser

def parse_arguments():
    parser = OptionParser()
    parser.add_option(
        "-o", "--output",
        help="output path for final parquet files. Default is here ('.')",
        dest="output_path",
        type="str",
        metavar="path/for/output",
        default="."
    )
    parser.add_option(
        "-t", "--tree-name",
        help="name of ttree in root files. Defaults to 'HTauTauTree'",
        dest="tree_name",
        type="str",
        metavar="path/for/output",
        default="HTauTauTree"
    )

    options, files = parser.parse_args()

    return options, files

def convert_to_parquet(filename, outpath=None, **kwargs):
    # get name of file without the directories
    file_base = os.path.basename(filename)
    # remove file extension (e.g. .root)
    file_base = ".".join(file_base.split(".")[:-1])

    # build new output file name
    output_filename = os.path.join(outpath, ".".join([file_base, "parquet"]))

    # load name of TTree
    tree_name = kwargs.get("tree_name")

    # if there is no tree name to load, throw error
    if not tree_name:
        raise ValueError("Must have tree name to open root files!")
    
    # open .root file with uproot
    up_file = up.open(f"{filename}:{tree_name}")

    # save ttree as parquet file with awkward
    ak.to_parquet(up_file.arrays(), output_filename)

def main(*files, **kwargs):
    output_path = kwargs.get("output_path", ".")

    # if the output path does not exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pbar = tqdm(files)
    for f in pbar:
        pbar.set_description(f"Converting file '{f}'")
        convert_to_parquet(f, outpath=output_path, **kwargs)


if __name__ == '__main__':
    options, files = parse_arguments()
    print(type(options))
    print(type(vars(options)))
    main(*files, **vars(options))