from omrdatasettools import Downloader, OmrDataset
from omrdatasettools import HomusImageGenerator

downloader = Downloader()
downloader.download_and_extract_dataset(OmrDataset.Homus_V2, "data")

HomusImageGenerator.create_images(raw_data_directory="data",
                                  destination_directory="homus_data",
                                  stroke_thicknesses=[3],
                                  canvas_width=96,
                                  canvas_height=192,
                                  staff_line_spacing=14,
                                  staff_line_vertical_offsets=[24])
