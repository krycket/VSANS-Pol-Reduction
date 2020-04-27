import requests
import posixpath
import os
from hashlib import sha256
import argparse

def retrieve_NCNR_datafiles(path, localpath="datafiles", extension=None, check_signature=True, verbose=True):
    """ Get a listing of all the datafiles matching the extension in
    the specified path, and retrieve them locally if they do no exist here
    or if check_signature=True and the remote signature differs from the
    local sha256 """
    
    pathlist = posixpath.split(path)
    data = {'pathlist[]' : pathlist}
    raw_listing = requests.post("https://ncnr.nist.gov/ncnrdata/listftpfiles_new.php", data=data).json()
    files_metadata = raw_listing['files_metadata']
    remote_url = "https://ncnr.nist.gov/pub/ncnrdata/" + posixpath.join(*raw_listing["pathlist"])

    if files_metadata == []:
        print("no files found in path {path}".format(path=path))
        return
    
    if extension is not None:
        files_metadata = dict([(fn, v) for fn, v in files_metadata.items() if fn.endswith(extension)])
        if len(files_metadata.values()) == 0:
            print("no files matching extension {extension} were found - exiting.".format(extension=extension))
            return

    if not os.path.exists(localpath):
        os.mkdir(localpath)
        
    for fn in files_metadata:
        retrieve = False
        local_fullpath = os.path.join(localpath, fn)
        if not os.path.exists(os.path.join(localpath, fn)):
            if verbose:
                print("file does not exist locally... retrieving: " + fn)
            retrieve = True
        else:
            if check_signature:
                local_hash = sha256(open(local_fullpath, 'rb').read()).hexdigest().upper()
                if local_hash.upper() != files_metadata[fn]['sha256'].upper():
                    retrieve = True
                    if verbose:
                        print(local_hash.upper(), files_metadata[fn]['sha256'].upper())
                        print("file exists locally but hash does not match remote, re-retrieving: " + fn)
                else:
                    if verbose:
                        print("file exists locally and hash matches remote: " + fn)
            else:
                if verbose:
                    print("file exists locally and not checking signatures: " + fn)
        
        if retrieve:
            file_contents = requests.get(posixpath.join(remote_url, fn)).content
            open(local_fullpath, 'wb').write(file_contents)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path in the ncnrdata repository you want to download from, e.g. vsans/202003/27322/data/")
    parser.add_argument("-l", "--localpath", type=str, default="datafiles", help="local path in which to store results (defaults to 'datafiles')")
    parser.add_argument("-e", "--extension", help="filter for file endings, e.g. .nxs.ngv")
    parser.add_argument("-f", "--force", action="store_true", help="force re-download even for files you already have")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress debugging printouts during execution")
    args = parser.parse_args()
    check_signature = (not args.force)
    verbose = (not args.quiet)
    print(args)

    retrieve_NCNR_datafiles(args.path, localpath=args.localpath, extension=args.extension, check_signature=check_signature, verbose=verbose)
