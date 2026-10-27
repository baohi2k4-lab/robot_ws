import csv

with open("logs/telemetry_2026-10-11_16-09-19.csv") as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r["FSM_State"] == "PATH_FOLLOWING"]
    print("PATH_FOLLOWING count:", len(rows))
    for r in rows[::10]:
        t = r["Time"]
        x = r["Pos_X_m"]
        y = r["Pos_Y_m"]
        yaw = float(r["Yaw_rad"]) * 57.3
        vx = r["Linear_Vel_mps"]
        wz = r["Angular_Vel_radps"]
        imu_yaw = float(r["IMU_Yaw_rad"]) * 57.3 if r["IMU_Yaw_rad"] else 0.0
        imu_wz = r["IMU_Angular_Vel_z"]
        print(f"t={t}: X={x}, Y={y}, Yaw={yaw:.1f}°, Vx={vx}, Wz={wz}, IMU_Yaw={imu_yaw:.1f}°, IMU_Wz={imu_wz}")
