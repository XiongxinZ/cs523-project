epoch = 1:90;
train_loss = [0.51539, 0.19882, 0.12011, 0.11993, 0.10020, 0.09187, 0.09715, 0.08659, 0.08803, 0.07701, ...
        0.07360, 0.07404, 0.06974, 0.06872, 0.06549, 0.06321, 0.06410, 0.06238, 0.05882, 0.05875, 0.05635, ...
        0.05271, 0.05009, 0.05315, 0.05218, 0.04687, 0.04582, 0.04629, 0.04431, 0.04342, 0.04385, ...
        0.04164, 0.04544, 0.04161, 0.03883, 0.03757, 0.03648, 0.03568, 0.03610, 0.03556, 0.03612, 0.03392, ...
        0.03401, 0.03417, 0.03429, 0.03219, 0.03347, 0.03190, 0.02949, 0.03177, 0.03023, 0.03362, 0.03320, ...
        0.03022, 0.02806, 0.02713, 0.02724, 0.02701, 0.02660, 0.02673, 0.02632, 0.02640, 0.02470, 0.02426, ...
        0.02561, 0.02555, 0.02409, 0.02504, 0.02552, 0.02646, 0.02545, 0.02398, 0.02409, 0.02293, 0.02343, ...
        0.02279, 0.02194, 0.02214, 0.02153, 0.02230, 0.02125, 0.02101, 0.02167, 0.02125, 0.01982, 0.02097, ...
        0.02043, 0.02108, 0.02073, 0.01987];
val_score = [0.01074,0.05463,0.08598,0.12492,0.14373,0.16971,0.16029,0.16746,0.17061,0.16344,0.18312,0.2013,0.1809, ...
        0.18735,0.1827,0.20463,0.16791,0.19971,0.18672,0.20193,0.2073,0.19521,0.23643,0.21537,0.20463,0.23418, ...
        0.26016,0.21762,0.25074,0.23955,0.28479,0.27357,0.28254,0.29643,0.29103,0.29238,0.31119,0.28656,0.28701, ...
        0.30984,0.30984,0.34656,0.32865,0.35238,0.32775,0.36627,0.34611,0.36717,0.35643,0.34137,0.37479,0.31299, ...
        0.33402,0.41178,0.3918,0.40836,0.42939,0.46029,0.46254,0.43119,0.43521,0.45312,0.5127,0.5409,0.48135,...
        0.51045,0.50148,0.49836,0.46164,0.42984,0.50061,0.46836,0.5673,0.5373,0.58926,0.56643,0.56103,0.55566,...
        0.64299,0.57447,0.59598,0.57939,0.57492,0.59283,0.69852,0.60552,0.63000,0.60045,0.6273,0.64119];
figure(2)
plot(epoch, train_loss, 'b');
legend('training loss');
figure(2)
plot(epoch, val_score, 'r')
legend('composite dice score on val');
xlabel('epoch');
xlabel('epoch');
