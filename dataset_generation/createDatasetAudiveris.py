from omrdatasettools import Downloader, OmrDataset
from omrdatasettools import AudiverisOmrImageGenerator

downloader = Downloader()
downloader.download_and_extract_dataset(OmrDataset.Audiveris, "audiveris")
audiveris = AudiverisOmrImageGenerator()
audiveris.extract_symbols(raw_data_directory="audiveris",
                                           destination_directory="audiveris_data")