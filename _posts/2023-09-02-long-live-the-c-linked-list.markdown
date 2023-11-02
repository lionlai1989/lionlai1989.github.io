---
layout: single
classes: wide
title: "Long Live the C: Linked List"
description: "."
toc: true
wip: true
date: 2023-09-02
categories: C
---

A linked list is a collection of objects that are stored in non-contiguous memory
locations.

## Linus Torvalds and "Good Taste"

In Linus Torvalds' TED talk
["The Mind Behind Linux,"](https://youtu.be/o8NPllzkFhE?feature=shared) he shares his
perspective on what constitutes "good taste" in software engineering. To illustrate his
point, he uses an example involving the deletion of a node in a singly linked list. In
this post, I aim to comprehend and implement the example Linus Torvalds provided in
order to cultivate a sense of "good taste" as an engineer 😀.

Before diving into the detail of deleting a node in a linked list, let's establish some
boilerplate code for creating and printing linked lists in C.

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int val;
    struct Node *next;
} Node;

Node *build_list(int arr[], int size) {
    if (size == 0) {
        return NULL;
    }

    Node *head = NULL;
    Node *curr = NULL;

    for (size_t i = 0; i < size; ++i) {
        Node *new_node = (Node *)malloc(sizeof(Node));
        new_node->val = arr[i];
        new_node->next = NULL;

        if (head == NULL) {
            head = new_node;
            curr = new_node;
        } else {
            curr->next = new_node;
            curr = new_node;
        }
    }
    return head;
}

void print_list(Node *head) {
    while (head != NULL) {
        printf("%d ", head->val);
        head = head->next;
    }
    printf("\n");
}
```

We can use this code to build and print a singly linked list:

```c
// gcc linux_linked_list.c && ./a.out
int main(void) {
    int arr[5] = {1, 2, 3, 4, 5}; // 5 integers
    int size = sizeof(arr) / sizeof(arr[0]);  // 5 = 20/4
    Node *head = build_list(arr, size);
    print_list(head);
    return 0;
}
```

The output will be:

```
1 2 3 4 5
```

Now we are ready to proceed to the deletion of a node.

### Deleting a Node in a Singly Linked List: The Bad Taste

The image below is shared in Linus Torvalds' TED talk and represents the pseudocode for
deleting a node in a singly linked list.

<div style="text-align:center">
  <img src="/assets/images/2023-09-02/remove_entry_bad_taste.png" width="800">
  <p style="font-size: 20px; color: #777;">Deleting a node in a list with bad taste.</p>
  <i></i>
</div>

We can implement the pseudocode above in C as follows:

```c
Node* remove_entry_bad_taste(Node *head, Node *entry) {
    Node *prev = NULL;
    Node *walk = head;

    // Walk the list to find `walk==entry`.
    while (walk != entry) {
        prev = walk;
        walk = walk->next;
    }

    // Remove the entry by updating `head` or `prev`
    if (prev == NULL) {  // The first node
        head = walk->next;
    } else {  // Not the first node
        prev->next = walk->next;
    }
    return head;
}
```

This code demonstrates a **common** approach to deleting a node from a singly linked
list. Note that my implementation of `remove_entry_bad_taste` returns the `head` because
in the case where the first node is being deleted, the pointer in the calling function
must also be updated.

However, Linus emphasizes that this approach lacks "good taste" because of the fact that
the if-else statement **must** handle the special case where the node to be deleted is
the first node.

We can test the code with the following:

```c
printf("Remove 3.\n");
head = remove_entry_bad_taste(head, head->next->next);  // Remove `3`.
print_list(head);
printf("Remove 1.\n");
head = remove_entry_bad_taste(head, head);  // Remove `1`.
print_list(head);
```

The output is:

```c
Remove 3.
1 2 4 5
Remove 1.
2 4 5
```

It works as expected, but Linus suggests there's a better way to do this - a way that
eliminates the need to handle the special case.

### Deleting a Node in a Singly Linked List: The Good Taste

The image below is also shared by Linus and represents the pseudocode for a more elegant
approach to deleting a node from a singly linked list.

<div style="text-align:center">
  <img src="/assets/images/2023-09-02/remove_entry_good_taste.png" width="800">
  <p style="font-size: 20px; color: #777;">Deleting a node in a list with good taste.</p>
  <i></i>
</div>

Here, the `indirect` pointer points to the `head` pointer and traverses the list to find
the `entry`. Once it locates the `entry`, it simply points to the next node of the
`entry`. Here is the implementation:

```c
void remove_entry_good_taste(Node **head, Node *entry) {
    // The "indirect" pointer points to the *address*
    // of the thing we'll update.
    Node **indirect = head;

    // Walk the list, looking for the thing that
    // points to the entry we want to remove.
    while (*indirect != entry) {
        indirect = &((*indirect)->next);
    }
    // .. and just remove it.
    *indirect = entry->next;
}
```

The key difference in this "good taste" approach is the removal of the if-else block and
the elimination of the need to return the head pointer. Instead, we pass a pointer
pointing to the head pointer as a double pointer (Node \*\*head).

To verify the code, we can use the following:

```c
int arr[5] = {1, 2, 3, 4, 5};
int size = sizeof(arr) / sizeof(arr[0]);  // 5 = 20/4
Node *head = build_list(arr, size);
print_list(head);
printf("Remove 3.\n");
remove_entry_good_taste(&head, head->next->next);  // Remove `3`.
print_list(head);
printf("Remove 1.\n");
remove_entry_good_taste(&head, head);  // Remove `1`.
print_list(head);
```

The output is:

```c
Remove 3.
1 2 4 5
Remove 1.
2 4 5
```

The output is exactly the same as in the "bad taste" example, but the code is more
elegant because it eliminates the need to deal with the special case.

**Notes:** I haven't discussed two edge cases: when the singly linked list is empty and
when the node to be removed does not exist in the list. These cases are essential to
handle in practical implementations but were not covered in this example.
