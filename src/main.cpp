#include <bits/stdc++.h>
#include <getopt.h>
using namespace std;

static inline string to_lower(const string &s){
    string out;
    out.reserve(s.size());
    for(char c: s) out.push_back(tolower((unsigned char)c));
    return out;
}

static inline string normalize_token(const string &s){
    string out;
    for(char c: s){
        if(isalpha((unsigned char)c) || c=='\'') out.push_back(tolower((unsigned char)c));
        else if(isdigit((unsigned char)c)) out.push_back(c);
        else out.push_back(' ');
    }
    return out;
}

vector<string> tokenize_and_process(const string &text, const unordered_set<string> &stopwords, const unordered_map<string,string> &lemmas){
    string norm = normalize_token(text);
    vector<string> tokens;
    string cur;
    for(char c: norm){
        if(isspace((unsigned char)c)){
            if(!cur.empty()){
                auto itl = lemmas.find(cur);
                string tk = (itl!=lemmas.end() ? itl->second : cur);
                if(!tk.empty() && stopwords.find(tk)==stopwords.end()) tokens.push_back(tk);
                cur.clear();
            }
        } else cur.push_back(c);
    }
    if(!cur.empty()){
        auto itl = lemmas.find(cur);
        string tk = (itl!=lemmas.end() ? itl->second : cur);
        if(!tk.empty() && stopwords.find(tk)==stopwords.end()) tokens.push_back(tk);
    }
    return tokens;
}

string read_file_all(const string &path){
    ifstream in(path);
    if(!in) return string();
    stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

unordered_set<string> load_stopwords(const string &path){
    unordered_set<string> s;
    if(path.empty()) return s;
    ifstream in(path);
    string w;
    while(in >> w){
        s.insert(to_lower(w));
    }
    return s;
}

unordered_map<string,string> load_lemmas(const string &path){
    unordered_map<string,string> m;
    if(path.empty()) return m;
    ifstream in(path);
    string a,b;
    while(in >> a >> b){
        m[to_lower(a)] = to_lower(b);
    }
    return m;
}

int main(int argc, char **argv){
    string dir;
    string stopfile;
    string lemmafile;
    bool help=false;

    int opt;
    while((opt = getopt(argc, argv, "d:s:l:h")) != -1){
        switch(opt){
            case 'd': dir = optarg; break;
            case 's': stopfile = optarg; break;
            case 'l': lemmafile = optarg; break;
            case 'h': help = true; break;
            default: help = true; break;
        }
    }
    if(help){
        cerr << "Uso:\n  " << argv[0] << " -d <dir_de_documentos> -s <stopwords.txt> -l <lemmas.txt>\n\n";
        cerr << "Si no usa -d debe pasar rutas de documentos .txt como argumentos posicionales.\n";
        return 1;
    }

    vector<string> doc_paths;
    if(!dir.empty()){
        // read all .txt files in dir
        // Portable simple approach using <filesystem>
        namespace fs = std::filesystem;
        try{
            for(auto &p: fs::directory_iterator(dir)){
                if(!p.is_regular_file()) continue;
                if(p.path().extension()!=".txt") continue;
                // ignore stopwords/lemmas files if they are inside the same dir
                string fname = p.path().filename().string();
                if(!stopfile.empty()){
                    if(p.path()==std::filesystem::path(stopfile) || fname==std::filesystem::path(stopfile).filename().string()) continue;
                }
                if(!lemmafile.empty()){
                    if(p.path()==std::filesystem::path(lemmafile) || fname==std::filesystem::path(lemmafile).filename().string()) continue;
                }
                doc_paths.push_back(p.path().string());
            }
            sort(doc_paths.begin(), doc_paths.end());
        } catch(exception &e){
            cerr << "Error leyendo directorio: " << e.what() << "\n";
            return 2;
        }
    } else {
        for(int i = optind; i < argc; ++i) doc_paths.push_back(argv[i]);
    }

    if(doc_paths.empty()){
        cerr << "No se encontraron documentos. Use -d <dir> o pase rutas .txt como argumentos.\n";
        return 3;
    }

    auto stopwords = load_stopwords(stopfile);
    auto lemmas = load_lemmas(lemmafile);

    int N = (int)doc_paths.size();
    vector<unordered_map<string,int>> doc_counts(N);
    unordered_map<string,int> df; // document frequency

    for(int i=0;i<N;++i){
        string content = read_file_all(doc_paths[i]);
        auto tokens = tokenize_and_process(content, stopwords, lemmas);
        for(const auto &t: tokens) doc_counts[i][t]++;
        // update df
        for(const auto &kv: doc_counts[i]) df[kv.first]++;
    }

    // vocabulary index
    vector<string> vocab;
    vocab.reserve(df.size());
    for(const auto &kv: df) vocab.push_back(kv.first);
    sort(vocab.begin(), vocab.end());
    unordered_map<string,int> term_index;
    for(size_t i=0;i<vocab.size();++i) term_index[vocab[i]] = (int)i;

    int V = (int)vocab.size();
    vector<vector<double>> idf(V);
    vector<double> idf_val(V);
    for(int i=0;i<V;++i){
        int dfi = df[vocab[i]];
        // avoid division by zero
        if(dfi<=0) idf_val[i] = 0.0;
        else idf_val[i] = log((double)N / (double)dfi);
    }

    // prepare TF-IDF vectors (sparse representation)
    vector<unordered_map<int,double>> tfidf_vec(N);
    vector<double> norms(N, 0.0);

    for(int i=0;i<N;++i){
        for(const auto &kv: doc_counts[i]){
            const string &term = kv.first;
            int tf = kv.second; // raw count
            int idx = term_index[term];
            double idfv = idf_val[idx];
            double tfidf = tf * idfv;
            tfidf_vec[i][idx] = tfidf;
            norms[i] += tfidf * tfidf;
        }
        norms[i] = sqrt(norms[i]);
    }

    // Output per-document tables
    for(int i=0;i<N;++i){
        cout << "Documento: " << doc_paths[i] << "\n";
        cout << "Idx\tTermino\tTF\tIDF\tTF-IDF\n";
        // build a vector of entries for sorting by index
        vector<pair<int,string>> entries;
        for(const auto &kv: doc_counts[i]) entries.push_back({term_index[kv.first], kv.first});
        sort(entries.begin(), entries.end());
        for(const auto &e: entries){
            int idx = e.first;
            const string &term = e.second;
            int tf = doc_counts[i].at(term);
            double idfv = idf_val[idx];
            double tfidf = tf * idfv;
            cout << idx << "\t" << term << "\t" << tf << "\t" << idfv << "\t" << tfidf << "\n";
        }
        cout << "\n";
    }

    // Cosine similarity matrix
    cout << "Matriz de similitud coseno:\n";
    // header
    cout << "\t";
    for(int j=0;j<N;++j) cout << j << "\t";
    cout << "\n";
    for(int i=0;i<N;++i){
        cout << i << "\t";
        for(int j=0;j<N;++j){
            double dot = 0.0;
            // iterate smaller map
            if(tfidf_vec[i].size() < tfidf_vec[j].size()){
                for(const auto &kv: tfidf_vec[i]){
                    auto it = tfidf_vec[j].find(kv.first);
                    if(it!=tfidf_vec[j].end()) dot += kv.second * it->second;
                }
            } else {
                for(const auto &kv: tfidf_vec[j]){
                    auto it = tfidf_vec[i].find(kv.first);
                    if(it!=tfidf_vec[i].end()) dot += kv.second * it->second;
                }
            }
            double denom = norms[i] * norms[j];
            double sim = 0.0;
            if(denom>0.0) sim = dot / denom;
            cout << sim << "\t";
        }
        cout << "\n";
    }

    return 0;
}
