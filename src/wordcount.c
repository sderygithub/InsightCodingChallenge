//
//  Copyright 2015 Sebastien Dery.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <dirent.h>
#include <pthread.h>

#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000

// Small project but hey! why not
int debug_mode = 2;

// I/O Folders
char *output_file_median = "wc_output/med_result.txt";
char *output_file = "wc_output/wc_result.txt";
char *input_dir = "wc_input";

// Should be adjusted to whatever number is in the expected corpus
const int vocab_hash_size = 10000000;

// Maximum number of values we intend to use for our median estimate (safety)
long long wordcount_max_size = 10000, wordcount_size = 0;
long long vocab_max_size = 10000, vocab_size = 0;
long long filequeue_max_size = 100, file_queue_size = 0;

// Will hold information about single words
struct vocab_word {
  long long cn;
  char *word;
};

struct file {
  char *filename;
};

struct file *file_queue;

// Word Frequency
struct vocab_word *vocab;
long long train_words = 0;
int *vocab_hash;

// Median
double *word_counts;
double *running_median;

// ===== ReadLine =====
// Read a single line from a file
// Supporting three different word tokenizer (Space, Tab, NewLine)
// Will skip punctuation marks
// ====================
double ReadLine(FILE *fin) {
  int inword = 0, ch;
  int word_count = 0;
  while (!feof(fin)) {
    // Case insensitive
    ch = tolower(fgetc(fin));
    // All the punctuations mark we are currently ignoring
    if ((ch == 13) || (ch == '.') || (ch == ',') || (ch == '(') || (ch == ')') || (ch == '-')) continue;
    // Supported word token separators
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (inword) {
        inword = 0;
        word_count++;
      }
      // Escape condition
      if (ch == '\n') break;
    }
    else inword = 1;
  }
  // Special case to include last word of a file
  if (feof(fin)) word_count++;
  return word_count;
}

// ===== ReadLine =====
// Reads a single word from a file
// Supporting three different word tokenizer (Space, Tab, NewLine)
// Will skip punctuation marks
// ====================
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    // Case insensitive    
    ch = tolower(fgetc(fin));
    // All the punctuations mark we are currently ignoring    
    if ((ch == 13) || (ch == '.') || (ch == ',') || (ch == '(') || (ch == ')') || (ch == '-')) continue;
    // Supported word token separators
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      // If at the beginning, simply ignore
      if (a == 0) continue;
      else break;
    }
    word[a] = ch;
    a++;
    // Truncate too long words
    if (a >= MAX_STRING - 1) a--;
  }
  word[a] = 0;
}

// ===== GetWordHash =====
// Returns hash value of a word
// =======================
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// ===== SearchVocab =====
// Returns position of a word in the vocabulary
// If the word is not found, returns -1
// =======================
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// ===== ReadWordIndex =====
// Reads a word and returns its index in the vocabulary
// =========================
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// ===== AddWordToVocab =====
// Adds a word to the vocabulary
// ==========================
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// ===== AddFileToQueue =====
// Adds a file to the file queue
// ==========================
int AddFileToQueue(char *filename) {
  unsigned int length = strlen(filename) + 1;
  file_queue[file_queue_size].filename = (char *)calloc(length, sizeof(char));
  strcpy(file_queue[file_queue_size].filename, filename);
  file_queue_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    filequeue_max_size += 1000;
    file_queue = (struct file *)realloc(file_queue, filequeue_max_size * sizeof(struct file));
  }
  return file_queue_size;
}

// ===== WordFrequencyCompare =====
// Sorting words by frequency
// ================================
int WordFrequencyCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// ===== AlphabeticalCompare =====
// Alphabetical sorting
// ===============================
int AlphabeticalCompare(const void *a, const void *b) {
  return strcmp(((struct vocab_word *)a)->word, ((struct vocab_word *)b)->word);
}

// ===== FilenameCompare =====
// FilenameCompare alphabetical sorting
// ===========================
int FilenameCompare(const void *a, const void *b) {
  return strcmp(((struct file *)a)->filename, ((struct file *)b)->filename);
}

// ===== CountCompare =====
// Number comparison
// ========================
int CountCompare(const void *a, const void *b) {
    return (*(double *)a) - (*(double *)b);
}

// ===== SortVocab =====
// Sorts the vocabulary in alphabetical order
// =====================
void SortVocab() {
  int a, size;
  unsigned int hash;
  // QuickSort the vocabulary
  qsort(&vocab[0], vocab_size, sizeof(struct vocab_word), AlphabeticalCompare);

  // Necessary to recompute hashing for potential further use
  train_words = 0;
  size = vocab_size;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < size; a++) {
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
    train_words += vocab[a].cn;
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// ===== SaveMedianResult =====
// Saves the current median result in the output file 
// ============================
void SaveMedianResult() {
  long long i;
  FILE *fo = fopen(output_file_median, "wb");
  for (i = 0; i < wordcount_size; i++) fprintf(fo, "%2.1f\n", running_median[i]);
  fclose(fo);
}

// ===== SaveWordFrequencyResult =====
// Saves the current word frequency result in the output file 
// ===================================
void SaveWordFrequencyResult() {
  long long i;
  FILE *fo = fopen(output_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

// ===== ComputeWordFrequency =====
// Computing loop for word frequency
// ================================
void ComputeWordFrequency(char *filename) {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  fin = fopen(filename, "rb");
  if (fin == NULL) {
    printf("ERROR: corpus file not found!\n");
    exit(1);
  }
  while (1) {
    // Read single word until End-Of-File
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    // Search for word in vocabulary and update count
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
  }
  SortVocab();
  fclose(fin);
}

// ===== RunningMedian =====
// Computing loop for running median
// =========================
void RunningMedian(char *filename) {
  // File check and operation 
  FILE *fin;
  fin = fopen(filename, "rb");
  if (fin == NULL) {
    printf("    ERROR: corpus file not found!\n");
    exit(1);
  }
  int i;
  int pair_line = 0;
  while (1) {
    // Read single lines until End-Of-File
    word_counts[wordcount_size] = ReadLine(fin);

    // Debugging
    //for (i = 0; i < wordcount_size+1; i++) printf("%2.1f,", word_counts[i]);
    //printf("\n");

    // Sort according to the number of words on a line
    qsort(&word_counts[0], wordcount_size+1, sizeof(double), CountCompare);

    // Compute the median
    //   Initial step: Pick the only available value
    //   Keep track of the line's parity so we don't need to perform 
    //   modulus or rounding operations. Might be beneficial, investigating this ;)
    if (wordcount_size == 0) running_median[wordcount_size] = word_counts[wordcount_size];
    else if (pair_line) running_median[wordcount_size] = (word_counts[(wordcount_size-1)/2] + word_counts[(wordcount_size-1)/2+1]) / 2;
    else running_median[wordcount_size] = word_counts[wordcount_size/2];

    // Increment update
    wordcount_size++;
    pair_line = !pair_line;

    // Debugging
    //for (i = 0; i < wordcount_size; i++) printf("%2.1f\n", word_counts[i]);

    // Escape condition
    if (feof(fin)) break;
  }

  // 
  fclose(fin);
}

int main(int argc, char **argv) {
  long long a, i;
  char filename_qfd[100];
  char new_name_qfd[100];

  printf("\n");
  printf("Word Count toolkit v 0.1\n");
  printf("  Sebastien Dery - 2015\n");
  printf("  sderymail [at] gmail [dot] com\n");
  printf("\n");

  // Initiate file queue for sorting
  file_queue = (struct file *)calloc(filequeue_max_size, sizeof(struct file));
  // Initiate vocabulary structure
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  // Initiate hash structure
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  // Initiate streaming structure
  word_counts = (double *)calloc(wordcount_max_size, sizeof(double));
  running_median = (double *)calloc(wordcount_max_size, sizeof(double));

  // Playing with file in C is tricky as it's mostly platform dependent
  // and thus would probably not be my language of choice for this task.
  // As a coding exercice on the other hand it's very nice to review basic concepts :)
  DIR *d;
  char *firstoccurence;
  char fullpath[1024];
  struct dirent *dir;
  d = opendir(input_dir);
  if (d) {
    while ((dir = readdir(d)) != NULL) {
      firstoccurence = strstr(dir->d_name,".txt");
      if (NULL != firstoccurence) {
        // Build filepath
        char *slash = "/";
        strcpy(fullpath, input_dir);
        strcat(strcat(fullpath, slash), dir->d_name);
        // Start computation
        AddFileToQueue(fullpath);
        printf("    %s added to queue ...\n",fullpath);
      }
    }
    closedir(d);

    // Sort file queue in alphabetical order
    qsort(&file_queue[0], file_queue_size, sizeof(double), FilenameCompare);
    printf("    Sorting file queue ...\n");

    for (i=0; i<file_queue_size; i++) {
        printf("    Streaming file %s ...\n",file_queue[i].filename);
        printf("      Compute word frequency ...\n");
        ComputeWordFrequency(file_queue[i].filename);
        printf("      Compute running median ...\n");
        RunningMedian(file_queue[i].filename);
    }
  }
  else {
    printf("ERROR: No input file could be found in the %s directory\n",input_dir);
  }

  if (debug_mode > 0) {
    printf("    Number of words in corpus: %lld\n", train_words);    
    printf("    Number of learned vocabulary: %lld\n", vocab_size);
  }
 
  // 
  SaveWordFrequencyResult();

  if (debug_mode > 0) printf("    Updated median is: %2.1f\n", running_median[wordcount_size-1]);
  
  //
  SaveMedianResult();

  // Cosmetic print
  printf("\n");

  return 0;
}
