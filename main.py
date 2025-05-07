def main():
    # === 1. WASTE CLASSIFICATION ===
    image_dir = "data/waste_images"  # folder with class subfolders
    train_ds, val_ds, class_names = load_image_dataset(image_dir)
    model = build_resnet50_model((224, 224, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=5)

    # Assume evaluation and label prediction for confusion matrix:
    # Replace with your actual y_true and y_pred values
    y_true = [0]*13880 + [1]*10825  # example class labels
    y_pred = [0]*13325 + [1]*555 + [0]*433 + [1]*10392
    plot_confusion_matrix(y_true, y_pred, labels=class_names)

    # === 2. ZONE ASSIGNMENT AND DENSITY VISUALIZATION ===
    gps_df = load_gps_data("data/litter_locations.csv")
    gps_df_filtered = filter_points_within_boundary(gps_df, "data/austin_boundary.geojson")
    compute_elbow(gps_df_filtered[['lat', 'lon']], max_k=6)
    clustered = assign_zones(gps_df_filtered.copy(), n_clusters=3)

    # Example waste density matrix and DataFrame
    waste_matrix = np.array([[120, 80, 50, 90], [100, 60, 70, 110], [140, 100, 60, 80]])
    df_heatmap = pd.DataFrame(waste_matrix, columns=['Plastic', 'Organic', 'Metal', 'Paper'], index=['Zone 1', 'Zone 2', 'Zone 3'])
    visualize_zone_distribution(df_heatmap)
    plot_waste_distribution_per_zone(df_heatmap)

    # === 3. WASTE FORECASTING ===
    ts_df = load_time_series("data/daily_waste.csv")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ts_df.values)
    X, y = create_sequences(scaled_data)
    split = int(len(X)*0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    model_lstm = build_lstm_model(X_train.shape[1:])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train, y_train, epochs=10, validation_split=0.2)
    y_pred_scaled = model_lstm.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred_scaled)
    plot_prediction_vs_actual(y_test_inv, y_pred_inv)
    plot_error_metrics(y_test_inv, y_pred_inv)

if __name__ == "__main__":
    main()