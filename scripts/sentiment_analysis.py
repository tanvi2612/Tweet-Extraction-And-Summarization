from imports import *
import pickle
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

def get_tweet_sentiment(tweet): 
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(tweet) 
    # set sentiment 
    if analysis.sentiment.polarity > 0: 
        # return 'Positive'
        return 1
    elif analysis.sentiment.polarity == 0: 
        # return 'Neutral'
        return 2
    else: 
        # return 'Negative'
        return 0

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

    
def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"

def string_to_int(sentiment):
    if sentiment == "Negative":
        return 0
    elif sentiment == "Neutral":
        return 2
    else:
        return 1

class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        # self.input_ids = token_ids
        # self.attention_mask = attention_mask
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self, item):
        # print("hello")
        # print(self.attention_mask[item])
        review = str(self.reviews[item])
        target = self.targets[item]
        # encoding = {'input_ids' : self.input_ids[item], 'attention_mask' : self.attention_mask[item] }
        encoding = self.tokenizer.encode_plus(review, max_length=100, add_special_tokens=True,       pad_to_max_length=True, return_token_type_ids=False,  return_attention_mask=True, return_tensors='pt')
        # print("hmm")
        # print(len(encoding['attention_mask'][0]))
        return {
        'review_text': review,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, ):
    MAX_LEN = 160
    BATCH_SIZE = 16

    # tokenID=[]
    # AttentionMask=[]
    # tokenList=[]
    # for row in df.index:
    #     tokens = tokenizer.tokenize(df["Text"][row])
    #     tokenList.append(tokens)
    #     encoding = tokenizer.encode_plus(df["Text"][row], max_length=60, add_special_tokens=True, pad_to_max_length=True, return_token_type_ids=False, padding=True, return_attention_mask=True, return_tensors='pt')
    #     tokenID.append(encoding['input_ids'][0].flatten())
    #     AttentionMask.append(encoding['attention_mask'][0].flatten())
        

    # df["TokenID"] = tokenID
    # df["AttentionMask"] = AttentionMask
    # df["tokenList"] = tokenList
    ds = GPReviewDataset(reviews=df.Text.to_numpy(), targets=df.Sentiment.to_numpy(),  tokenizer=tokenizer, max_len=MAX_LEN)
    return DataLoader( ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=True), df

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

def train_epoch( model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model( input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model( input_ids=input_ids, attention_mask=attention_mask )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

  
def main(): 
    tweets = []
    with open("../dataset/raw-dataset/raw") as f:
        lines = f.readlines()
        for l in lines:
            tweet = json.loads(l)
            # print(tweet.keys())
            text, _, _, _ = clean(tweet["full_text"])
            senti = get_tweet_sentiment(text)
            # print(tweet.keys())
            tweets.append([tweet["id_str"], text, senti])
            # break
    df = pd.DataFrame(data=tweets, columns=["ID", "Text", "Sentiment"])



    #---------------------------TEXTBLOB----------------------------

    
    print("Sentiment Analysis using Text Blob")
    # picking positive tweets from tweets 
    ptweet = df[df['Sentiment'] == 1]
    ptweets = ptweet["Text"].tolist()
    # percentage of positive tweets 
    print("Positive tweets percentage: {} %".format(round(100*len(ptweets)/len(tweets), 2))) 
    # picking negative tweets from tweets 
    ntweet = df[df['Sentiment'] == 0]
    ntweets = ntweet["Text"].tolist()
    # percentage of negative tweets 
    print("Negative tweets percentage: {} %".format(round(100*len(ntweets)/len(tweets), 2))) 
    # percentage of neutral tweets 
    print("Neutral tweets percentage: {} %".format(round(100*(len(tweets) -(len( ntweets )+len( ptweets)))/len(tweets), 2))) 
    # printing first 5 positive tweets 
    # print("\n\nPositive tweets:") 
    # for tweet in ptweets[:20]: 
    #     print(tweet) 
    # # printing first 5 negative tweets 
    # print("\n\nNegative tweets:") 
    # for tweet in ntweets[:20]: 
    #     print(tweet) 
    # dataset = df
    # print(dataset["Text"])

    
    #--------------------------------------Training Models-------------------------------------------------------
    

    # print("Training Models using the above data")

    tf_vector = get_feature_vector(np.array(df["Text"]).ravel())
    X = tf_vector.transform(np.array(df["Text"]).ravel())
    y = np.array(df["Sentiment"]).ravel()
    

    # Training Naive Bayes model
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)
    # NB_model = MultinomialNB()
    # NB_model.fit(X_train, y_train)
    # y_predict_nb = NB_model.predict(X_test)
    # print("Accuracy in Naive Bayes:", accuracy_score(y_test, y_predict_nb))
    
    # # Training Logistics Regression model
    # LR_model = LogisticRegression(solver='lbfgs', max_iter=2000)
    # LR_model.fit(X_train, y_train)
    # y_predict_lr = LR_model.predict(X_test)
    # print("Accucracy in Logistic Regression Model:", accuracy_score(y_test, y_predict_lr))

    # exit(1)
    #--------------------BERT-------------------
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=30)
    df_val, df_test = train_test_split(df_test, test_size=0.1, random_state=30)

    sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    train_data_loader, df_train = create_data_loader(df_train, tokenizer)
    val_data_loader, df_val = create_data_loader(df_val, tokenizer)
    test_data_loader, df_test = create_data_loader(df_test, tokenizer)
    token_lens = []
    for txt in df.Text:
        tokens = tokenizer.encode(txt, max_length=512, truncation=True)
        token_lens.append(len(tokens))

    sns.displot(token_lens)
    plt.xlim([0, 150])
    plt.xlabel('Token count')
    plt.show()
    exit()
    class_names = ['negative', 'neutral', 'positive']

    model = SentimentClassifier(len(class_names))
    model = model.to(device)


    EPOCHS = 10
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=0, num_training_steps=total_steps )
    loss_fn = nn.CrossEntropyLoss().to(device)


    history = defaultdict(list)
    best_accuracy = 0
    f = open("training.txt", "a")  
    for epoch in range(EPOCHS):        
        f.write(f'Epoch {epoch + 1}/{EPOCHS}\n')
        f.write('-' * 10)
        f.write("\n")
        train_acc, train_loss = train_epoch( model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train) )
        f.write(f'Train loss {train_loss} accuracy {train_acc}\n')
        val_acc, val_loss = eval_model( model, val_data_loader, loss_fn, device, len(df_val) )
        f.write(f'Val loss {val_loss} accuracy {val_acc}\n')
        f.write('\n')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
    f.close()

    f = open('train_stat', 'wb')
    pickle.dump(history, f)
    f.close() 

    test_acc, _ = eval_model( model, test_data_loader, loss_fn, device, len(df_test))

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)

    print(classification_report(y_test, y_pred, target_names=class_names))
    
if __name__ == "__main__": 
    # calling main function 
    main() 
    