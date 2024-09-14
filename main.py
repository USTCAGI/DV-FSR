import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset,load_dataset_seq
from client import FedRecClient,FedRecSequentialClient,FedRecBert4RecClient,FedRecSASRecClient
from server import FedRecServer,FedRecSequentialServer
from attack import AttackClient, BaselineAttackClient,PipAttackClient,SeqAttackClient,BaselineSeqAttackClient
from attack_my_method import SeqAttackClient_method1,SeqAttackClient_method2,SeqAttackClient_method2_1,SeqAttackClient_method1_3,SeqAttackClient_method_pipattack,SeqAttackClient_method1_2_3

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)

    t0 = time()
    if args.model_type == "NCF":
        m_item, all_train_ind, all_test_ind, items_popularity = load_dataset(args.path + args.dataset)
    elif args.model_type == "SASrec" or args.model_type == "BERT4rec" or args.model_type == "SASrec2" or args.model_type == "Bert4rec2":
        m_item, all_train_ind, all_test_ind, items_popularity = load_dataset_seq(args.path + args.dataset)
    _, target_items = torch.Tensor(-items_popularity).topk(1)
    target_items = target_items.tolist()  # Select the least popular item as the target item
    if args.model_type == "NCF":
        server = FedRecServer(m_item, args.dim, eval(args.layers),items_popularity).to(args.device)
    # elif args.model_type == "SASrec":
    #     server = FedRecSequentialServer(m_item,args.dim,"SASRec").to(args.device)
    # elif args.model_type == "BERT4rec":
    #     server = FedRecSequentialServer(m_item,args.dim,"BERT4Rec").to(args.device)
    elif args.model_type == "SASrec2":
        server = FedRecSequentialServer(m_item,args.dim,"SASrec2").to(args.device)
    elif args.model_type == "Bert4rec2":
        server = FedRecSequentialServer(m_item,args.dim,"Bert4rec2").to(args.device)
    clients = []
    if args.model_type == "NCF":
        for train_ind, test_ind in zip(all_train_ind, all_test_ind):
            clients.append(
                FedRecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
            )
    # elif args.model_type == "SASrec":
    #     for train_ind, test_ind in zip(all_train_ind, all_test_ind):
    #         clients.append(
    #             FedRecSequentialClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
    #         )
    # elif args.model_type == "BERT4rec":
    #     for train_ind, test_ind in zip(all_train_ind, all_test_ind):
    #         clients.append(
    #             FedRecBert4RecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
    #         )
    elif args.model_type == "SASrec2":
        for train_ind, test_ind in zip(all_train_ind, all_test_ind):
            clients.append(
                FedRecSASRecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
            )
    elif args.model_type == "Bert4rec2":
        for train_ind, test_ind in zip(all_train_ind, all_test_ind):
            clients.append(
                FedRecBert4RecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
            )
    malicious_clients_limit = int(len(clients) * args.clients_limit)
    #new_array = np.random.choice(all_train_ind,size = malicious_clients_limit,replace = False)
    #new_array = random.sample(all_train_ind,malicious_clients_limit)
    indices = random.sample(range(len(all_train_ind)), malicious_clients_limit)
    selected_train_ind = [all_train_ind[i] for i in indices]
    selected_test_ind = [all_test_ind[i] for i in indices]
    # if args.attack == 'A-ra' or args.attack == 'A-hum':
    #     for _ in range(malicious_clients_limit):
    #         clients.append(AttackClient(target_items, m_item, args.dim).to(args.device))
    if args.attack == 'EB':
        for _ in range(malicious_clients_limit):
            clients.append(BaselineAttackClient(target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'RA':
        for _ in range(malicious_clients_limit):
            #train_ind = [i for i in target_items]
            for __ in range(args.items_limit - len(target_items)):
                item = np.random.randint(m_item)
                while item in train_ind:
                    item = np.random.randint(m_item)
                train_ind.append(item)
            train_ind.append(target_items[0])
            #clients.append(BaselineAttackClient(train_ind, m_item, args.dim).to(args.device))
            clients.append(BaselineSeqAttackClient(train_ind, m_item, args.dim).to(args.device))

            #clients.append(FedRecSASRecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'pipattack':
        for i in range(malicious_clients_limit):
            clients.append(
                #PipAttackClient(target_items, m_item, args.dim,server.popularity_model).to(args.device)
                SeqAttackClient_method_pipattack(selected_test_ind[i],target_items, m_item, args.dim,selected_train_ind[i]).to(args.device)
            )
    elif args.attack == 'seqAttack':
        for i in range(malicious_clients_limit):
            clients.append(
                SeqAttackClient(selected_test_ind[i],target_items, m_item, args.dim,selected_train_ind[i]).to(args.device)
            )
    elif args.attack == 'method1':
        for i in range(malicious_clients_limit):
            clients.append(
                SeqAttackClient_method1(selected_test_ind[i],target_items, m_item, args.dim,selected_train_ind[i]).to(args.device)
            )
    elif args.attack == 'method2':
        for i in range(malicious_clients_limit):
            clients.append(
                SeqAttackClient_method2(selected_test_ind[i],target_items, m_item, args.dim,selected_train_ind[i]).to(args.device)
            )
    elif args.attack == 'method_C-FSR':
        for i in range(malicious_clients_limit):
            clients.append(
                SeqAttackClient_method2_1(selected_test_ind[i],target_items, m_item, args.dim,selected_train_ind[i]).to(args.device)
            )
    elif args.attack == 'method_S-FSR':
        for i in range(malicious_clients_limit):
            clients.append(
                SeqAttackClient_method1_3(selected_test_ind[i],target_items, m_item, args.dim,selected_train_ind[i]).to(args.device)
            )
    elif args.attack == 'method_DV-FSR':
        for i in range(malicious_clients_limit):
            clients.append(
                SeqAttackClient_method1_2_3(selected_test_ind[i],target_items, m_item, args.dim,selected_train_ind[i]).to(args.device)
            )
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("Target items: %s." % str(target_items))
    print("output format: ({HR@10, Prec@10, NDCG@10}), ({ER@5, ER@10, ER@20, ER@30})")

    # Init performance
    t1 = time()
    test_result, target_result = server.eval_(clients)
    print("Iteration 0(init), (%.7f, %.7f, %.7f) on test" % tuple(test_result) +
          ", (%.7f, %.7f, %.7f, %.7f) on target." % tuple(target_result) +
          " [%.1fs]" % (time() - t1))

    try:
        lr = args.lr
        for epoch in range(1, args.epochs + 1):
            if epoch % 5 == 0:
                #lr = lr * (0.1 ** (epoch // 10))
                if args.agg == "RFA":
                    #lr = lr * 0.1 #RFA聚合衰减策略
                    #lr = lr * 0.2
                    lr = lr * 0.15
                    #lr = lr * 0.125
                elif args.agg == "common":
                    lr = lr * 0.5
                elif args.agg == "mixagg":
                    if args.model_type == "SASrec2":
                        lr = lr * 0.5
                    elif args.model_type == "Bert4rec2":
                        lr = lr * 0.15
                lr = round(lr,10)
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []
            print("lr为",lr)
            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                loss = server.train_(clients, batch_clients_idx,lr)
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()

            t2 = time()
            test_result, target_result = server.eval_(clients)
            print("Iteration %d, loss = %.5f [%.1fs]" % (epoch, total_loss, t2 - t1) +
                  ", (%.7f, %.7f, %.7f) on test" % tuple(test_result) +
                  ", (%.7f, %.7f, %.7f, %.7f) on target." % tuple(target_result) +
                  " [%.1fs]" % (time() - t2))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    #setup_seed(20220110)
    #setup_seed(20240809)
    setup_seed(20000000)
    main()
