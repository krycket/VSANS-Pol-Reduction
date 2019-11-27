import requests
import posixpath
import os
from hashlib import sha256


def retrieve_NCNR_datafiles(path, local_path="datafiles", extension=None, check_signature=True, verbose=True):
    """ Get a listing of all the datafiles matching the extension in
    the specified path, and retrieve them locally if they do no exist here
    or if check_signature=True and the remote signature differs from the
    local sha256 """
    
    pathlist = posixpath.split(path)
    data = {'pathlist[]' : pathlist}
    raw_listing = requests.post("https://ncnr.nist.gov/ncnrdata/listftpfiles_new.php", data=data).json()
    files_metadata = raw_listing['files_metadata']
    remote_url = "https://ncnr.nist.gov/pub/ncnrdata/" + posixpath.join(*raw_listing["pathlist"])
    
    if not os.path.exists(local_path):
        os.mkdir(local_path)
        
    for fn in files_metadata:
        retrieve = False
        local_fullpath = os.path.join(local_path, fn)
        if not os.path.exists(os.path.join(local_path, fn)):
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
