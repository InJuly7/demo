#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    char *argv[2];
    argv[0] = "cat";
    argv[1] = NULL;  // 使用NULL而不是0更清晰
    
    pid_t pid = fork();
    
    if (pid == -1) {
        // fork失败
        perror("fork failed");
        exit(1);
    }
    else if (pid == 0) {
        // 子进程
        close(0);  // 释放标准输入的文件描述符
        
        // 打开文件并检查是否成功
        int fd = open("input.txt", O_RDONLY);
        if (fd == -1) {
            perror("open input.txt failed");
            exit(1);
        }
        
        // 此时input.txt的文件描述符为0（标准输入）
        // 即标准输入重定向到input.txt
        
        // 执行cat命令，cat从stdin(0)读取，输出到stdout(1)
        execv("/bin/cat", argv);  // 使用execv并指定完整路径
        
        // 如果exec成功，下面的代码不会执行
        perror("exec failed");
        exit(1);
    }
    else {
        // 父进程
        int status;
        wait(&status);  // 等待子进程结束
        
        if (WIFEXITED(status)) {
            printf("Child process exited with status: %d\n", WEXITSTATUS(status));
        }
    }
    
    return 0;
}
