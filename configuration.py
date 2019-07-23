# -%-coding:utf-8-%-
import argparse
config = argparse.ArgumentParser()
config.add_argument('--db_name', default='ml-100k', help='Database Name')
config.add_argument('--similarity', default='cosine', help='The way to construct laplacian')
config.add_argument('--type', default='item_based', help='Item_based or User_based')
config.add_argument('--min_common', default=2, type=int, help='Minimum common ratings for each pair of users/movies')
config.add_argument('--filename', default='ua', help='Splited Dataset Filename' )

FLAGS, unparsed = config.parse_known_args()


