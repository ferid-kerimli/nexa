package com.nexa.ai.tokenizer;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SimpleTokenizerTest {

    @Test
    void shouldAddTokensToVocabulary() {

        Vocabulary vocabulary = new Vocabulary();

        int javaId = vocabulary.addToken("Java");
        int springId = vocabulary.addToken("Spring");

        assertEquals(2, javaId);
        assertEquals(3, springId);

        assertEquals("Java", vocabulary.getToken(javaId));
        assertEquals("Spring", vocabulary.getToken(springId));
    }

    @Test
    void shouldTokenizeText() {

        Vocabulary vocabulary = new Vocabulary();

        vocabulary.addToken("Java");
        vocabulary.addToken("is");
        vocabulary.addToken("powerful");

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(vocabulary);

        List<String> tokens =
                tokenizer.tokenize("Java is powerful");

        assertEquals(
                List.of("Java", "is", "powerful"),
                tokens
        );
    }

    @Test
    void shouldEncodeText() {

        Vocabulary vocabulary = new Vocabulary();

        vocabulary.addToken("Java");
        vocabulary.addToken("is");
        vocabulary.addToken("powerful");

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(vocabulary);

        List<Integer> ids =
                tokenizer.encode("Java is powerful");

        assertEquals(
                List.of(2, 3, 4),
                ids
        );
    }

    @Test
    void shouldDecodeTokenIds() {

        Vocabulary vocabulary = new Vocabulary();

        vocabulary.addToken("Java");
        vocabulary.addToken("is");
        vocabulary.addToken("powerful");

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(vocabulary);

        String text =
                tokenizer.decode(List.of(2, 3, 4));

        assertEquals(
                "Java is powerful",
                text
        );
    }

    @Test
    void unknownTokenShouldUseUnk() {

        Vocabulary vocabulary = new Vocabulary();

        vocabulary.addToken("Java");

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(vocabulary);

        List<Integer> ids =
                tokenizer.encode("Python");

        assertEquals(
                List.of(0),
                ids
        );
    }

    @Test
    void shouldHandleMultipleSpaces() {

        Vocabulary vocabulary = new Vocabulary();

        vocabulary.addToken("Java");
        vocabulary.addToken("is");
        vocabulary.addToken("powerful");

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(vocabulary);

        List<String> tokens =
                tokenizer.tokenize("Java    is     powerful");

        assertEquals(
                List.of("Java", "is", "powerful"),
                tokens
        );
    }

    @Test
    void shouldSeparatePunctuation() {

        Vocabulary vocabulary = new Vocabulary();

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(vocabulary);

        List<String> tokens =
                tokenizer.tokenize("Hello, world!");

        assertEquals(
                List.of("Hello", ",", "world", "!"),
                tokens
        );
    }

    @Test
    void shouldHandleNumbers() {

        Vocabulary vocabulary = new Vocabulary();

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(vocabulary);

        List<String> tokens =
                tokenizer.tokenize("NEXA 2026");

        assertEquals(
                List.of("NEXA", "2026"),
                tokens
        );
    }

    @Test
    void shouldHandleQuestionAndExclamationMarks() {

        Vocabulary vocabulary = new Vocabulary();

        SimpleTokenizer tokenizer =
                new SimpleTokenizer(vocabulary);

        List<String> tokens =
                tokenizer.tokenize("How are you?!");

        assertEquals(
                List.of(
                        "How",
                        "are",
                        "you",
                        "?",
                        "!"
                ),
                tokens
        );
    }
}