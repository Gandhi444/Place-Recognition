import requests


def test(index,distance):
    student_id = index  # TODO: put your student id here
    distance_name = distance  # supported values are: manhattan, euclidean, cosine
    with open('results.pickle', 'rb') as file:
        predictions = file.read()

    response = requests.post(f'https://zpo.dpieczynski.pl/{student_id}', headers={'distance': distance_name},
                             data=predictions)
    if response.status_code == 200:
        print(response.json())
    else:
        print(response.status_code)
        print(response.text)
