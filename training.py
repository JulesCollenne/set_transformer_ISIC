from matplotlib import pyplot as plt


class Training:

    def __init__(self, train_gen, val_gen, test_gen):
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen

    def run(self):
        n_run = 4
        # models = ("BYOL", "Moco", "SimCLR", "SimSiam", "SwaV")
        models = ["reduced_CNN"]

        do_training = True
        do_testing = True
        visualize_preds = False

        plt.figure(figsize=(8, 6))

        les_best_thresholds = {
            "BYOL": [0.490, 0.490, 0.469, 0.469],
            "Moco": [0.510, 0.531, 0.510, 0.531],
            "SimCLR": [0.429, 0.490, 0.408, 0.388],
            "SimSiam": [0.510, 0.531, 0.510, 0.490],
            "SwaV": [0.551, 0.469, 0.490, 0.571],
        }

        for model_name in models:
            print("Training:", model_name)

            num_inds = 20
            n_classes = 2
            dim_inner_output = 10
            patience = 15

            n_features = sum(['feature' in col for col in pd.read_csv(f"features/{model_name}_val.csv").columns])

            all_auc, all_bacc, all_sens, all_spec = [], [], [], []
            all_tpr, all_fpr = [], []
            base_fpr = np.linspace(0, 1, 101)

            class_weights = torch.tensor([0.02, 0.98]).cuda()
            criterion = nn.CrossEntropyLoss(class_weights)

            for run in range(n_run):
                outname = f"{model_name}/residual_{model_name}_{run}_{num_inds}_{dim_inner_output}.pth"
                # outname = f"{model_name}/residual_{model_name}_{run}_{num_inds}.pth"
                if do_training:
                    model = ResConvSet(n_features, num_inds, n_classes, num_inds=num_inds, num_heads=args.n_heads,
                                       dim_inner_output=dim_inner_output)
                    train(model, train_gen, val_gen, criterion, outname, patience)

                if do_testing:
                    model = ResConvSet(n_features, num_inds, n_classes, num_inds=num_inds, num_heads=args.n_heads,
                                       dim_inner_output=dim_inner_output)

                    model = nn.DataParallel(model)
                    model = model.cuda()

                    # checkpoint = torch.load(f"models/{model_name}/{model_name}.pth")
                    # model.load_state_dict(checkpoint["state_dict"])
                    model.load_state_dict(torch.load(f"models/{outname}"))
                    model.eval()

                    losses, total, correct, true_labels, predicted_probs = do_one_epoch(val_gen, model, criterion)

                    avg_loss, avg_acc = np.mean(losses), correct / total

                    auc = roc_auc_score(true_labels, predicted_probs)
                    best_bacc = 0
                    best_thresh = 0
                    thresholds = np.linspace(0, 1, 50)

                    for thresh in thresholds:
                        balanced_acc = balanced_accuracy_score(true_labels,
                                                               (np.array(predicted_probs) > thresh).astype(int))
                        if balanced_acc > best_bacc:
                            best_bacc = balanced_acc
                            best_thresh = thresh

                    print(
                        f"val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f}"
                        f" val balanced acc {best_bacc:.3f} best threshold {best_thresh:.3f}")

                    losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
                    # for imgs, lbls in test_gen.train_data():
                    for imgs, lbls in test_gen.train_data():
                        imgs = torch.Tensor(imgs).cuda()
                        lbls = torch.Tensor(lbls).long().cuda()
                        preds = model(imgs)

                        zero_rows_mask = torch.all(imgs == 0, dim=2)
                        non_zero_rows_mask = ~zero_rows_mask
                        preds = preds[non_zero_rows_mask]
                        lbls = lbls[non_zero_rows_mask]

                        loss = criterion(preds.view(-1, 2), lbls.view(-1))

                        losses.append(loss.item())
                        total += lbls.view(-1).shape[0]
                        correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

                        true_labels += lbls.view(-1).cpu().numpy().tolist()
                        predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()

                    avg_loss, avg_acc = np.mean(losses), correct / total

                    data = pd.DataFrame({'p': [pred for pred in predicted_probs],
                                         'target': true_labels})
                    # np.savetxt(f"predictions/{model_name}_{run}_{num_inds}", np.array(predicted_probs), delimiter=",")
                    data.to_csv(f"predictions/{model_name}_{run}_{num_inds}.csv", index=False)
                    binary_predictions = (np.array(predicted_probs) > best_thresh).astype(int)

                    conf_matrix = confusion_matrix(true_labels, binary_predictions)

                    tn, fp, fn, tp = conf_matrix.ravel()
                    sensitivity = tp / (tp + fn)
                    specificity = tn / (tn + fp)

                    all_auc.append(roc_auc_score(true_labels, predicted_probs))
                    all_bacc.append(balanced_accuracy_score(true_labels, binary_predictions))
                    all_sens.append(sensitivity)
                    all_spec.append(specificity)
                    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
                    tpr = np.interp(base_fpr, fpr, tpr)
                    tpr[0] = 0.0
                    all_tpr.append(tpr)
                    all_fpr.append(fpr)

                    print(
                        f"test loss {avg_loss:.3f} test acc {avg_acc:.3f} test AUC {all_auc[-1]:.3f}"
                        f" test balanced acc {all_bacc[-1]:.3f}")

                if visualize_preds:
                    data = pd.read_csv(f"predictions/{model_name}_{run}_{num_inds}.csv")
                    true_labels = data["target"]
                    predicted_probs = data["p"]

                    best_thresh = les_best_thresholds[model_name][run]

                    binary_predictions = (np.array(predicted_probs) > best_thresh).astype(int)

                    conf_matrix = confusion_matrix(true_labels, binary_predictions)

                    tn, fp, fn, tp = conf_matrix.ravel()
                    sensitivity = tp / (tp + fn)
                    specificity = tn / (tn + fp)

                    all_auc.append(roc_auc_score(true_labels, predicted_probs))
                    all_bacc.append(balanced_accuracy_score(true_labels, binary_predictions))
                    all_sens.append(sensitivity)
                    all_spec.append(specificity)
                    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
                    tpr = np.interp(base_fpr, fpr, tpr)
                    tpr[0] = 0.0
                    all_tpr.append(tpr)
                    all_fpr.append(fpr)

            if visualize_preds:
                print(model_name)
                print(
                    f"test AUC {np.mean(all_auc):.3f} +/- {np.std(all_auc)} "
                    f"test bacc {np.mean(all_bacc):.3f} +/- {np.std(all_bacc)} "
                    f"test sensitivity {np.mean(all_sens):.3f} +/- {np.std(all_sens)} "
                    f"test specificity {np.mean(all_spec):.3f} +/- {np.std(all_spec)}")

                plot_average_roc(all_tpr, base_fpr, all_auc, model_name)

        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Average ROC Curve with Standard Deviation')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig('roc_curve_with_std.png', dpi=300, bbox_inches='tight')
