import os
import zipfile
import urllib.request

def download_orekit_data():
    """Download and extract Orekit data files."""
    # Create orekit-data directory if it doesn't exist
    if not os.path.exists('orekit-data'):
        os.makedirs('orekit-data')
    
    # Download URL for orekit-data
    url = "https://gitlab.orekit.org/orekit/orekit-data/-/archive/master/orekit-data-master.zip"
    
    print("Downloading Orekit data...")
    try:
        # Download the file
        urllib.request.urlretrieve(url, "orekit-data-master.zip")
        
        # Extract the zip file
        print("Extracting Orekit data...")
        with zipfile.ZipFile("orekit-data-master.zip", 'r') as zip_ref:
            # Extract all files
            for member in zip_ref.namelist():
                # Skip the top-level directory and extract contents directly to orekit-data
                if member.startswith('orekit-data-master/') and len(member) > len('orekit-data-master/'):
                    target = os.path.join('orekit-data', member[len('orekit-data-master/'):])
                    
                    # Create directories if needed
                    if member.endswith('/'):
                        os.makedirs(target, exist_ok=True)
                    else:
                        # Make sure parent directory exists
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        
                        # Extract file
                        with zip_ref.open(member) as source, open(target, 'wb') as target_file:
                            target_file.write(source.read())
        
        # Clean up
        os.remove("orekit-data-master.zip")
        print("Orekit data downloaded and extracted successfully!")
        
    except Exception as e:
        print(f"Error downloading Orekit data: {e}")
        print("You can manually download from: https://gitlab.orekit.org/orekit/orekit-data")

# Alternative using orekit's built-in function if available
def download_orekit_data_alternative():
    """Alternative method using orekit's built-in downloader."""
    try:
        import orekit
        from orekit.pyhelpers import download_orekit_data_curdir
        download_orekit_data_curdir()
        print("Orekit data downloaded successfully using built-in function!")
    except Exception as e:
        print(f"Built-in download failed: {e}")
        print("Trying manual download...")
        download_orekit_data()

if __name__ == "__main__":
    download_orekit_data_alternative()