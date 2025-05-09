#!/usr/bin/env python3
import os, sys, time, datetime, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from pump_spectrometer_test import initialize_pumps
from adafruit import setUpSerialPort, setWhiteLightBrightness, setIntegrationTime, setGain
from spectrometer import initialize_spectrometer
from spectrometer_data import read_spectrometer

opt_method = "bayesian"
BASELINE_FLOW = 0.1
MAX_ADDITIONAL_FLOW = 0.4
MIN_FLOW_DELTA = 0.005
MAX_FLOW_DELTA = 0.01
DEADBAND = 0.2
K_i = 0.2
K_DELTA = 0.5
K_INT = 0.1
xi = 0.01
MAX_HISTORY = 50
REAL_DELAY = 5
data_dir = "./logs"
CAL_REF = 804.0
CAL_ACID = 0.0
CAL_BASIC = 0.11427
PH_ACID = 1.6
PH_BASIC = 13.0
EPSILON = 1e-3
SIMULATOR_MODE = False

def scale_pH(x):
    return (x - 1) / 13.0

def unscale_pH(x_scaled):
    return x_scaled * 13.0 + 1

data_log = []
absorbance_data = {}

def compute_absorbance(I_sample, I_ref):
    I_sample = max(I_sample, 1e-9)
    I_ref = max(I_ref, 1e-9)
    return np.log10(I_ref / I_sample)

def get_current_pH_details():
    data, _ = read_spectrometer()
    if data is None or 590 not in data:
        return None, None
    I_590 = data[590]
    A_unknown = compute_absorbance(I_590, CAL_REF)
    if abs(CAL_BASIC - CAL_ACID) < 1e-12:
        return None, None
    alpha_cal = (A_unknown - CAL_ACID) / (CAL_BASIC - CAL_ACID)
    pH = PH_ACID + alpha_cal * (PH_BASIC - PH_ACID)
    abs_spectrum = {wl: compute_absorbance(data[wl], CAL_REF) for wl in data}
    return {"pH": pH, "I_590": I_590, "A_unknown": A_unknown, "raw_data": data}, abs_spectrum

def log_iteration(iteration, details, current_dose_flow, target_pH,
                  rec_acid_flow, rec_base_flow, reward, penalty, gp_confidence):
    entry = {
        "timestamp": time.time(),
        "iteration": iteration,
        "target_pH": target_pH,
        "measured_pH": details["pH"],
        "current_dose_flow": current_dose_flow,
        "recommended_acid_flow": rec_acid_flow,
        "recommended_base_flow": rec_base_flow,
        "error": details["pH"] - target_pH,
        "reward": reward,
        "penalty": penalty,
        "gp_confidence": gp_confidence,
        "raw_data": json.dumps(details["raw_data"])
    }
    data_log.append(entry)

def save_log_data():
    os.makedirs(data_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(data_dir, f"ph_control_{now}.csv")
    pd.DataFrame(data_log).to_csv(filename, index=False)

def plot_data(iterations, measured_pH, expected_pH, target_pH):
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, measured_pH, marker='o', linestyle='-', color='blue', label="Measured pH")
    plt.plot(iterations, expected_pH, marker='x', linestyle='--', color='green', label="Expected pH")
    plt.axhline(target_pH, color='red', linestyle='--', label=f"Target pH ({target_pH})")
    plt.xlabel("Iteration")
    plt.ylabel("pH")
    plt.title("pH vs. Iteration")
    plt.ylim(0, 14)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_absorbance():
    plt.figure(figsize=(8, 6))
    for it, spectrum in absorbance_data.items():
        wavelengths = sorted(spectrum.keys())
        abs_values = [spectrum[wl] for wl in wavelengths]
        plt.plot(wavelengths, abs_values, label=f"Iteration {it}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title("Cumulative Absorbance Spectra")
    plt.ylim(0, 1)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()

def update_gp_model(gp_model, X_data, y_data, new_input, new_output, max_history=MAX_HISTORY):
    new_X = np.array([[scale_pH(new_input)]])
    if X_data.size == 0:
        X_data = new_X
        y_data = np.array([new_output])
    else:
        X_data = np.vstack([X_data, new_X])
        y_data = np.append(y_data, new_output)
        if len(y_data) > max_history:
            X_data = X_data[-max_history:]
            y_data = y_data[-max_history:]
    gp_model.fit(X_data, y_data)
    return gp_model, X_data, y_data

def plot_training_data(gp_acid, gp_base):
    try:
        acid_X = gp_acid.X_train_
        acid_y = gp_acid.y_train_
        base_X = gp_base.X_train_
        base_y = gp_base.y_train_
        acid_X_unscaled = unscale_pH(acid_X.flatten()) if acid_X.max() <= 1.0 else acid_X.flatten()
        base_X_unscaled = unscale_pH(base_X.flatten()) if base_X.max() <= 1.0 else base_X.flatten()
        plt.figure(figsize=(8,6))
        plt.scatter(acid_X_unscaled, acid_y, color='blue', label='Acid Training Data')
        plt.scatter(base_X_unscaled, base_y, color='green', label='Base Training Data')
        plt.xlabel("Measured pH")
        plt.ylabel("Additional Flow (mL/min)")
        plt.title("Ideal Training Data from GP Models")
        plt.xlim(1,14)
        plt.legend()
        plt.show()
    except Exception as e:
        print("Training data not found in GP models.", e)

def visualize_updated_models(gp_acid, gp_base, acid_history, base_history):
    X_test = np.linspace(1,14,200).reshape(-1,1)
    X_test_scaled = (X_test - 1) / 13.0
    if gp_acid is not None:
        y_mean_acid, y_std_acid = gp_acid.predict(X_test_scaled, return_std=True)
        y_mean_acid = np.clip(y_mean_acid, 0, MAX_ADDITIONAL_FLOW)
        lower_acid = np.clip(y_mean_acid - 1.96 * y_std_acid, 0, MAX_ADDITIONAL_FLOW)
        upper_acid = np.clip(y_mean_acid + 1.96 * y_std_acid, 0, MAX_ADDITIONAL_FLOW)
        plt.figure(figsize=(10,6))
        plt.scatter(unscale_pH(acid_history[:,0]), acid_history[:,1], color='blue', label='Acid Training Data')
        plt.plot(X_test, y_mean_acid, 'r-', label='GP Mean Prediction (Acid)')
        plt.fill_between(X_test.flatten(), lower_acid, upper_acid, color='red', alpha=0.2, label='95% Confidence Interval')
        plt.xlabel("Measured pH")
        plt.ylabel("Additional Acid Flow (mL/min)")
        plt.title("Acid GP Model")
        plt.xlim(1,14)
        plt.ylim(0, MAX_ADDITIONAL_FLOW)
        plt.legend()
        plt.show()
    if gp_base is not None:
        y_mean_base, y_std_base = gp_base.predict(X_test_scaled, return_std=True)
        y_mean_base = np.clip(y_mean_base, 0, MAX_ADDITIONAL_FLOW)
        lower_base = np.clip(y_mean_base - 1.96 * y_std_base, 0, MAX_ADDITIONAL_FLOW)
        upper_base = np.clip(y_mean_base + 1.96 * y_std_base, 0, MAX_ADDITIONAL_FLOW)
        plt.figure(figsize=(10,6))
        plt.scatter(unscale_pH(base_history[:,0]), base_history[:,1], color='green', label='Base Training Data')
        plt.plot(X_test, y_mean_base, 'r-', label='GP Mean Prediction (Base)')
        plt.fill_between(X_test.flatten(), lower_base, upper_base, color='red', alpha=0.2, label='95% Confidence Interval')
        plt.xlabel("Measured pH")
        plt.ylabel("Additional Base Flow (mL/min)")
        plt.title("Base GP Model")
        plt.xlim(1,14)
        plt.ylim(0, MAX_ADDITIONAL_FLOW)
        plt.legend()
        plt.show()

def manual_calibration():
    input("Place the REFERENCE solution, press Enter...")
    ref_data, _ = read_spectrometer()
    if ref_data is None or 590 not in ref_data:
        sys.exit(1)
    ref_intensity = ref_data[590]
    input("Place the BASIC solution, press Enter...")
    basic_data, _ = read_spectrometer()
    if basic_data is None or 590 not in basic_data:
        sys.exit(1)
    basic_intensity = basic_data[590]
    input("Place the ACIDIC solution, press Enter...")
    acid_data, _ = read_spectrometer()
    if acid_data is None or 590 not in acid_data:
        sys.exit(1)
    acid_intensity = acid_data[590]
    CAL_REF_new = ref_intensity
    CAL_BASIC_new = np.log10(ref_intensity / basic_intensity)
    CAL_ACID_new = np.log10(ref_intensity / acid_intensity)
    if CAL_ACID_new < 0:
        CAL_ACID_new = 0.0
    cont = input("Continue? (Y/N): ").strip().upper()
    if cont == "Y":
        globals()['CAL_REF'] = CAL_REF_new
        globals()['CAL_BASIC'] = CAL_BASIC_new
        globals()['CAL_ACID'] = CAL_ACID_new
        return CAL_REF_new, CAL_ACID_new, CAL_BASIC_new
    else:
        sys.exit(0)

def run_mode():
    global CAL_REF, CAL_ACID, CAL_BASIC
    default_acid_X = np.hstack([scale_pH(np.array([[7], [10], [14]])), np.zeros((3, 1))])
    default_acid_y = np.array([0.1, 0.3, 0.5])
    default_base_X = np.hstack([scale_pH(np.array([[1], [4], [7]])), np.zeros((3, 1))])
    default_base_y = np.array([0.5, 0.3, 0.1])
    current_acid_data_X = default_acid_X.copy()
    current_acid_data_y = default_acid_y.copy()
    current_base_data_X = default_base_X.copy()
    current_base_data_y = default_base_y.copy()
    integrated_error = 0.0
    decay_factor = 0.9
    prev_additional = 0.0
    smooth_alpha = 0.05
    setUpSerialPort("/dev/cu.usbmodem14401")
    setIntegrationTime(500)
    setGain(8)
    setWhiteLightBrightness(2)
    initialize_spectrometer(integration_time=500, gain=8)
    if input("Perform manual calibration? (Y/N): ").strip().upper() == "Y":
        manual_calibration()
    setUpSerialPort("/dev/cu.usbmodem14401")
    setIntegrationTime(500)
    setGain(8)
    setWhiteLightBrightness(2)
    initialize_spectrometer(integration_time=500, gain=8)
    pumps = initialize_pumps()
    if 'a' not in pumps:
        sys.exit(1)
    machine_a = pumps['a']
    target_pH = float(input("Enter target pH: "))
    ACID_MODEL_FILE = os.path.join(data_dir, "gp_model_acid.pkl")
    BASE_MODEL_FILE = os.path.join(data_dir, "gp_model_base.pkl")
    try:
        gp_acid = joblib.load(ACID_MODEL_FILE)
        gp_base = joblib.load(BASE_MODEL_FILE)
        plot_training_data(gp_acid, gp_base)
    except:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        gp_acid = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10, optimizer_kwargs={'maxiter': 1000})
        gp_base = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10, optimizer_kwargs={'maxiter': 1000})
        gp_acid.fit(current_acid_data_X, current_acid_data_y)
        gp_base.fit(current_base_data_X, current_base_data_y)
        plot_training_data(gp_acid, gp_base)
    prev_rec_acid_flow = BASELINE_FLOW
    prev_rec_base_flow = BASELINE_FLOW
    iterations = []
    measured_pH_list = []
    expected_pH_list = []
    iteration_count = 0
    try:
        while True:
            iteration_count += 1
            iterations.append(iteration_count)
            details, spectrum = get_current_pH_details()
            if details is None:
                continue
            current_pH = details["pH"]
            measured_pH_list.append(current_pH)
            expected_pH_list.append(target_pH)
            absorbance_data[iteration_count] = spectrum
            error = current_pH - target_pH
            if abs(error) < DEADBAND:
                if iteration_count > 1:
                    rec_acid_flow = prev_rec_acid_flow
                    rec_base_flow = prev_rec_base_flow
                else:
                    rec_acid_flow = BASELINE_FLOW
                    rec_base_flow = BASELINE_FLOW
            else:
                if current_pH > target_pH:
                    scaled_input = np.array([[scale_pH(current_pH)]])
                    predicted = float(gp_acid.predict(scaled_input))
                    additional = predicted + K_i * error
                    additional = np.clip(additional, 0, MAX_ADDITIONAL_FLOW)
                    rec_acid_flow = BASELINE_FLOW + additional
                    rec_base_flow = BASELINE_FLOW
                else:
                    scaled_input = np.array([[scale_pH(current_pH)]])
                    predicted = float(gp_base.predict(scaled_input))
                    additional = predicted + K_i * abs(error)
                    additional = np.clip(additional, 0, MAX_ADDITIONAL_FLOW)
                    rec_base_flow = BASELINE_FLOW + additional
                    rec_acid_flow = BASELINE_FLOW
            delta_limit = MIN_FLOW_DELTA + (MAX_FLOW_DELTA - MIN_FLOW_DELTA) * min(abs(error) / 3.0, 1.0)
            if rec_acid_flow - prev_rec_acid_flow > delta_limit:
                rec_acid_flow = prev_rec_acid_flow + delta_limit
            if rec_base_flow - prev_rec_base_flow > delta_limit:
                rec_base_flow = prev_rec_base_flow + delta_limit
            prev_rec_acid_flow = rec_acid_flow
            prev_rec_base_flow = rec_base_flow
            machine_a.changePump(1)
            machine_a.rate(rec_acid_flow if current_pH > target_pH else BASELINE_FLOW)
            machine_a.run(1)
            machine_a.changePump(2)
            machine_a.rate(rec_base_flow if current_pH < target_pH else BASELINE_FLOW)
            machine_a.run(2)
            time.sleep(REAL_DELAY)
            log_iteration(iteration_count, details, BASELINE_FLOW, target_pH,
                          rec_acid_flow if current_pH > target_pH else BASELINE_FLOW,
                          rec_base_flow if current_pH < target_pH else BASELINE_FLOW,
                          1/(abs(error)+EPSILON), 0.0, {})
            plot_data(iterations, measured_pH_list, expected_pH_list, target_pH)
            plot_absorbance()
            if current_pH > target_pH:
                additional_to_update = rec_acid_flow - BASELINE_FLOW
                gp_acid, current_acid_data_X, current_acid_data_y = update_gp_model(
                    gp_acid, current_acid_data_X, current_acid_data_y, current_pH, additional_to_update)
            elif current_pH < target_pH:
                additional_to_update = rec_base_flow - BASELINE_FLOW
                gp_base, current_base_data_X, current_base_data_y = update_gp_model(
                    gp_base, current_base_data_X, current_base_data_y, current_pH, additional_to_update)
    except KeyboardInterrupt:
        machine_a.stop(1)
        machine_a.stop(2)
    finally:
        save_log_data()
        acid_train_file = os.path.join(data_dir, "gp_training_data_acid_mode7.csv")
        base_train_file = os.path.join(data_dir, "gp_training_data_base_mode7.csv")
        pd.DataFrame({'pH': current_acid_data_X[:,0], 'flow': current_acid_data_y}).to_csv(acid_train_file, index=False)
        pd.DataFrame({'pH': current_base_data_X[:,0], 'flow': current_base_data_y}).to_csv(base_train_file, index=False)
        try:
            acid_history = np.hstack([current_acid_data_X, current_acid_data_y.reshape(-1,1)]) if current_acid_data_X.size else np.empty((0,2))
            base_history = np.hstack([current_base_data_X, current_base_data_y.reshape(-1,1)]) if current_base_data_X.size else np.empty((0,2))
            visualize_updated_models(gp_acid, gp_base, acid_history, base_history)
        except Exception as e:
            print("Error visualizing updated models:", e)

if __name__ == "__main__":
    run_mode()
