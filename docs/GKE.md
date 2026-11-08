> Przeniesione z README 8.11.2026 (redakcja po feedbacku prowadzącego); treść bez zmian.

# GKE Autopilot — jednorazowe wdrożenie 22.10.2026

## Komendy (z Quickstartu)

**GKE Autopilot (stretch, płatne; projekt GCP z billingiem, gcloud z `gke-gcloud-auth-plugin`)**
— ten sam obraz i te same manifesty; obraz z Artifact Registry przypięty digestem i podmieniany
w locie (`k8s/deployment.yaml` bez zmian), test przez port-forward, bez publicznego adresu.
`<PROJECT>` / `<NUMER>` = ID i numer projektu GCP, `<DIGEST>` = digest z `docker push`:

```bash
gcloud services enable container.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create banking77 --repository-format=docker --location=europe-central2
gcloud auth configure-docker europe-central2-docker.pkg.dev
docker tag banking77-intent-api:distilbert-v1 europe-central2-docker.pkg.dev/<PROJECT>/banking77/banking77-intent-api:distilbert-v1
docker push europe-central2-docker.pkg.dev/<PROJECT>/banking77/banking77-intent-api:distilbert-v1
# tylko gdy konto usługi węzłów nie ma ról w projekcie (u nas: nowy projekt, zero ról)
gcloud artifacts repositories add-iam-policy-binding banking77 --location=europe-central2 --member=serviceAccount:<NUMER>-compute@developer.gserviceaccount.com --role=roles/artifactregistry.reader
gcloud container clusters create-auto banking77-gke --region=europe-central2
kubectl apply -f k8s/service.yaml
kubectl set image --local -f k8s/deployment.yaml api=europe-central2-docker.pkg.dev/<PROJECT>/banking77/banking77-intent-api@<DIGEST> -o yaml | kubectl apply -f -
kubectl rollout status deployment/banking77-intent-api --timeout=540s
kubectl port-forward svc/banking77-intent-api 18082:8000   # zostaw działające
uv run python scripts/smoke_api.py --url http://localhost:18082 --timeout 120   # drugi terminal
gcloud container clusters delete banking77-gke --region=europe-central2
gcloud artifacts repositories delete banking77 --location=europe-central2
```

## Przebieg i koszt (z sekcji Wdrożenie)

- **GKE (stretch #4, wykonane 22.10 ręcznie, poza CI):** ten sam obraz i te same manifesty na
  GKE Autopilot w `europe-central2` (Warszawa, Kubernetes 1.36.4-gke). Klaster żył ok. 14 min;
  `teardown` tego samego wieczoru — klaster i repozytorium Artifact Registry usunięte, w projekcie
  brak VM, dysków, adresów i reguł przekierowania. Przebieg (komendy w *Quickstart*; log z ID
  projektu zastąpionym przez `<PROJECT>`: [`docs/gke-2026-10-22.log`](gke-2026-10-22.log)):
  - obraz z czystego drzewa `fbf1627` (pliki modelu zgodne z manifestem Release); push do
    Artifact Registry 36 s — obraz amd64 821 775 937 B, wdrożony po digeście indeksu
    `sha256:c85a5c62…`;
  - `gcloud container clusters create-auto` 255 s; rollout 193 s od `kubectl apply`: Autopilot
    dołożył węzeł od zera, pobranie obrazu ok. 60 s, gotowość ok. 12–14 s po starcie kontenera —
    do tego czasu sonda startowa dostawała connection refused (uvicorn słucha dopiero po
    załadowaniu i rozgrzaniu modelu; w kontenerze lokalnie ok. 4 s);
  - smoke `[OK]` przez `kubectl port-forward` (`/health`, 14 zdań kontrolnych, 422, wersja
    `distilbert-v1`); 2/2 pody, 0 restartów, po 369 MiB RAM; Autopilot nie zmienił zasobów
    z manifestu (dodał tylko domyślne ephemeral-storage 1Gi);
  - konto usługi węzłów nie miało w nowym projekcie żadnych ról → `roles/artifactregistry.reader`
    tylko na repozytorium obrazu;
  - bez publicznego adresu: Service NodePort bez reguły zapory, test przez port-forward — trafia
    w jeden pod, więc nie sprawdza rozkładu ruchu przez Service.

  Koszt wg cennika GKE dla Warszawy (22.10): pody 0,0573 USD za vCPU·h i 0,0063421 USD za GiB·h
  (Autopilot liczy zasoby podów, nie węzła) → nasze 2 × 0,5 vCPU / 0,5 GiB ≈ 0,064 USD/h; opłatę
  za klaster 0,10 USD/h pokrywa kredyt GKE (74,40 USD/mies. na konto rozliczeniowe), **nie
  compute**; Artifact Registry: 0,5 GiB miesięcznie bez opłat. Rzeczywista kwota wg raportu
  rozliczeń GCP za październik 2026 (odczyt 28.10): koszt 0,10 zł, w całości pokryty środkami
  („inne oszczędności” −0,10 zł) — do zapłaty 0,00 zł.
Log z ID projektu zastąpionym przez `<PROJECT>`: [gke-2026-10-22.log](gke-2026-10-22.log).
