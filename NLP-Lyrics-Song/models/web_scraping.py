import requests
from bs4 import BeautifulSoup
import os
import re

with open("../api_key.txt", "r") as f:
    api_key = f.read().strip()

GENIUS_API_TOKEN = api_key


def request_artist_info(artist_name, page):
    base_url = "https://api.genius.com"
    headers = {"Authorization": "Bearer " + GENIUS_API_TOKEN}
    search_url = base_url + "/search?per_page=10&page=" + str(page)
    data = {"q": artist_name}
    response = requests.get(search_url, data=data, headers=headers)
    return response


def request_song_url(artist_name, song_cap):
    page = 1
    songs = []

    while True:
        response = request_artist_info(artist_name, page)
        json = response.json()

        song_info = []
        for hit in json["response"]["hits"]:
            if artist_name.lower() in hit["result"]["primary_artist"]["name"].lower():
                song_info.append(hit)

        for song in song_info:
            if len(songs) < song_cap:
                url = song["result"]["url"]
                songs.append(url)

        if len(songs) == song_cap:
            break
        else:
            page += 1

    print("Found {} songs by {}".format(len(songs), artist_name))
    return songs


def scrape_song_lyrics(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, "html.parser")
    lyrics = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-5 Dzxov").get_text()
    # remove identifiers like chorus, verse, etc
    lyrics = re.sub(r"[\(\[].*?[\)\]]", "", lyrics)
    # remove empty lines
    lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
    return lyrics


def write_lyrics_to_file(artist_name, song_count):
    with open("../lyrics/" + artist_name.lower() + ".txt", "w", encoding="utf-8") as f:
        urls = request_song_url(artist_name, song_count)
        for url in urls:
            lyrics = scrape_song_lyrics(url)
            f.write(lyrics)
    num_lines = sum(
        1
        for line in open(
            "../lyrics/" + artist_name.lower() + ".txt", "r", encoding="utf-8"
        )
    )
    print("Wrote {} lines to file from {} songs".format(num_lines, song_count))


def write_lyrics_to_file(artist_name, song_count):
    with open("../lyrics/" + artist_name.lower() + ".txt", "w", encoding="utf-8") as f:
        urls = request_song_url(artist_name, song_count)
        for url in urls:
            song_title = (
                url.split("/")[-1].replace("-", " ").title()
            )  # Extract song title from URL
            lyrics = scrape_song_lyrics(url)
            f.write(song_title + "\n")  # Write song title to file
            f.write(lyrics + "\n\n")  # Write lyrics to file
    num_lines = sum(
        1
        for line in open(
            "../lyrics/" + artist_name.lower() + ".txt", "r", encoding="utf-8"
        )
    )
    print("Wrote {} lines to file from {} songs".format(num_lines, song_count))


write_lyrics_to_file("Kendrick Lamar", 200)
write_lyrics_to_file("Drake", 200)
