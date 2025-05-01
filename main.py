from image_localizer import extract_netvlad, localize_image, retrieve_image_matches

def main():
    print("Extract NetVLAD features from an image:")
    print(extract_netvlad.__doc__)
    print()

    print("Localize an image:")
    print(localize_image.__doc__)
    print()

    print("Retrieve image matches:")
    print(retrieve_image_matches.__doc__)

if __name__ == "__main__":
    main()
