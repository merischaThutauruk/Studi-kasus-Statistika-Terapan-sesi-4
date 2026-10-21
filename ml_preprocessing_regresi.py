import math
import random


class HousePriceMLWorkflow:

    def __init__(self, n_samples=150, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        self.raw_data = []
        self.clean_data = []
        self.weights = []  

    def collect_raw_data(self):
        print("=== [1] DATA COLLECTION: SIMULASI DATA PROPERTI MENTAH ===")
        random.seed(self.seed)
        lokasi_opsi = [
            ("Pedesaan", 1),
            ("Pinggiran Kota", 2),
            ("Pusat Kota (Bandar)", 3),
        ]

        for i in range(self.n_samples):
            luas_m2 = round(random.uniform(45.0, 250.0), 1)
            kamar = random.randint(1, 5)
            nama_lok, skor_lok = random.choice(lokasi_opsi)

            noise = random.gauss(0, 25.0)
            harga_juta = (
                150.0
                + (luas_m2 * 4.5)
                + (kamar * 35.0)
                + (skor_lok * 120.0)
                + noise
            )

            if i % 12 == 0:
                luas_m2 = None
            if i % 19 == 0:
                kamar = None

            record = {
                "id_rumah": f"HOM-{100 + i}",
                "luas_m2": luas_m2,
                "kamar_tidur": kamar,
                "nama_lokasi": nama_lok,
                "skor_lokasi": skor_lok,
                "harga_juta": round(harga_juta, 2),
            }
            self.raw_data.append(record)

            if i == 15 or i == 50:
                self.raw_data.append(record.copy())

        print(f"- Berhasil mengumpulkan {len(self.raw_data)} baris data mentah.")
        print(
            "- Data masih mengandung Missing Values (None) dan Duplikasi Data.\n"
        )

    def preprocess_data(self):
        print(
            "=== [2] DATA PREPROCESSING: MENCEGAH 'GARBAGE IN, GARBAGE OUT' ==="
        )
        unique_records = {}
        duplikat_dihapus = 0
        for r in self.raw_data:
            if r["id_rumah"] not in unique_records:
                unique_records[r["id_rumah"]] = r
            else:
                duplikat_dihapus += 1

        temp_data = list(unique_records.values())

        valid_luas = [
            r["luas_m2"] for r in temp_data if r["luas_m2"] is not None
        ]
        valid_kamar = sorted(
            [
                r["kamar_tidur"]
                for r in temp_data
                if r["kamar_tidur"] is not None
            ]
        )

        mean_luas = round(sum(valid_luas) / len(valid_luas), 1)
        median_kamar = valid_kamar[len(valid_kamar) // 2]

        missing_luas_count = 0
        missing_kamar_count = 0
        for r in temp_data:
            if r["luas_m2"] is None:
                r["luas_m2"] = mean_luas
                missing_luas_count += 1
            if r["kamar_tidur"] is None:
                r["kamar_tidur"] = median_kamar
                missing_kamar_count += 1
            self.clean_data.append(r)

        print(f"- Duplikasi dihapus                   : {duplikat_dihapus} baris")
        print(
            f"- Imputasi Missing Value (Luas)       : {missing_luas_count} baris diisi dengan Mean ({mean_luas} m2)"
        )
        print(
            f"- Imputasi Missing Value (Kamar)      : {missing_kamar_count} baris diisi dengan Median ({median_kamar} kamar)"
        )
        print(
            f"- Total data bersih siap pemodelan    : {len(self.clean_data)} baris\n"
        )

    def split_and_train_model(self):
        print(
            "=== [3] DATA SPLITTING (80% TRAIN : 20% TEST) & MODEL TRAINING ==="
        )
        split_idx = int(0.8 * len(self.clean_data))
        train_set = self.clean_data[:split_idx]
        test_set = self.clean_data[split_idx:]

        print(
            f"- Jumlah Data Latih (Training Set)    : {len(train_set)} sampel (80%)"
        )
        print(
            f"- Jumlah Data Uji (Testing Set)       : {len(test_set)} sampel (20% - Unseen Data)"
        )

        b0, b1, b2, b3 = 100.0, 100.0, 10.0, 50.0
        learning_rate = 0.01
        epochs = 1500
        n = len(train_set)

        for _ in range(epochs):
            grad_b0 = grad_b1 = grad_b2 = grad_b3 = 0.0
            for row in train_set:
                x1 = row["luas_m2"] / 100.0
                x2 = float(row["kamar_tidur"])
                x3 = float(row["skor_lokasi"])
                y_true = row["harga_juta"]

                y_pred = b0 + (b1 * x1) + (b2 * x2) + (b3 * x3)
                err = y_pred - y_true

                grad_b0 += (2.0 / n) * err
                grad_b1 += (2.0 / n) * err * x1
                grad_b2 += (2.0 / n) * err * x2
                grad_b3 += (2.0 / n) * err * x3

            b0 -= learning_rate * grad_b0
            b1 -= learning_rate * grad_b1
            b2 -= learning_rate * grad_b2
            b3 -= learning_rate * grad_b3

        self.weights = [b0, b1 / 100.0, b2, b3]

        print("\n[PERSAMAAN REGRESI LINIER BERGANDA YANG DIPEROLEH MESIN]")
        print(
            f"  Harga (Juta) = {self.weights[0]:.2f} + ({self.weights[1]:.2f} * Luas_m2) "
            f"+ ({self.weights[2]:.2f} * Kamar) + ({self.weights[3]:.2f} * Skor_Lokasi)\n"
        )
        return train_set, test_set

    def predict(self, luas_m2, kamar, skor_lokasi):
        b0, b1, b2, b3 = self.weights
        return b0 + (b1 * luas_m2) + (b2 * kamar) + (b3 * skor_lokasi)

    def evaluate_metrics(self, dataset, label_set="Data"):
        y_true_list = [r["harga_juta"] for r in dataset]
        y_pred_list = [
            self.predict(r["luas_m2"], r["kamar_tidur"], r["skor_lokasi"])
            for r in dataset
        ]

        n = len(dataset)
        mean_y = sum(y_true_list) / n

        mae = sum(abs(yt - yp) for yt, yp in zip(y_true_list, y_pred_list)) / n
        mse = (
            sum((yt - yp) ** 2 for yt, yp in zip(y_true_list, y_pred_list)) / n
        )
        rmse = math.sqrt(mse)

        ss_tot = sum((yt - mean_y) ** 2 for yt in y_true_list)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true_list, y_pred_list))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        print(f"--- Metrik Evaluasi {label_set} ---")
        print(f"  > Mean Absolute Error (MAE)         : Rp {mae:.2f} Juta")
        print(f"  > Root Mean Squared Error (RMSE)    : Rp {rmse:.2f} Juta")
        print(
            f"  > Koefisien Determinasi (R-Squared) : {r_squared:.4f} ({r_squared * 100:.2f}%)"
        )
        return mae, rmse, r_squared

    def run_evaluation_and_inference(self, train_set, test_set):
        print(
            "=== [4] EVALUASI STATISTIKA TERAPAN & ANALISIS OVERFITTING ==="
        )
        train_mae, _, train_r2 = self.evaluate_metrics(
            train_set, "TRAINING SET (80%)"
        )
        print()
        test_mae, _, test_r2 = self.evaluate_metrics(
            test_set, "TESTING SET / UNSEEN DATA (20%)"
        )

        selisih_r2 = abs(train_r2 - test_r2)
        print("\n[DIAGNOSIS STATISTIK MODEL]")
        if train_r2 < 0.60 and test_r2 < 0.60:
            print(
                "  Status: UNDERFITTING (Model terlalu lemah mempelajari pola data)."
            )
        elif selisih_r2 > 0.15:
            print(
                f"  Status: OVERFITTING (Selisih R^2 {selisih_r2:.4f} terlalu jauh antara Train & Test)."
            )
        else:
            print(
                f"  Status: GOOD FIT / GENERALISASI OPTIMAL (Selisih R^2 hanya {selisih_r2:.4f})."
            )
            print(
                f"  Interpretasi Statistik: {test_r2 * 100:.1f}% variasi harga rumah mampu dijelaskan secara akurat oleh fitur luas, kamar, dan lokasi.\n"
            )

        print("=== [5] DEPLOYMENT INFERENCE: PREDIKSI RUMAH BARU ===")
        contoh_rumah = [
            {"luas": 120.0, "kamar": 3, "lokasi": "Pusat Kota (Bandar)", "skor": 3},
            {"luas": 70.0, "kamar": 2, "lokasi": "Pedesaan", "skor": 1},
        ]
        for c in contoh_rumah:
            pred = self.predict(c["luas"], c["kamar"], c["skor"])
            print(
                f"- Rumah Luas {c['luas']}m2, {c['kamar']} Kamar di {c['lokasi']:<20} -> Prediksi Harga: Rp {pred:,.2f} Juta"
            )


if __name__ == "__main__":
    workflow = HousePriceMLWorkflow()
    workflow.collect_raw_data()
    workflow.preprocess_data()
    train_data, test_data = workflow.split_and_train_model()
    workflow.run_evaluation_and_inference(train_data, test_data)