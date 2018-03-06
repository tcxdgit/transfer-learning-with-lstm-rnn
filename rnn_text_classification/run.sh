#!/bin/bash
IFS=$'\n\n'
#field=$1
#action=$2
source ../function.sh

function choose_action(){
    option=$(whiptail --title "命令选择" --clear --menu "请选择其中一个：" 12 35 3 \
         "test" "聊天测试" "find" "查找进程" "kill" "关闭进程" "restart" "重启进程" \
        3>&1 1>&2 2>&3)
    exitstatus=$?
    if [ $exitstatus = 0 ]; then
        echo $option
    else
        echo ''
    fi
}

function kill_process(){
    echo 'kill process'
    ids=$(get_pid 'mq_classify' ' '${field}' ')
    echo "kill mq_classify ${field}"
    
    if [[ ${ids} != '' ]]; then
        for id in ${ids}
        do
            echo 'kill '${id}
            kill ${id}
        done
    fi
}

field=$(choose_field)

if [[ ${field} != '' ]]; then
    action=$(choose_action)
fi

if [[ ${action} != '' ]]; then
    confirm=$(choose_confirm '场景：'$field' 命令：'$action' ')
fi

if [[ ${confirm} != '' ]]; then
    case $action in
        'test')
            echo ${field}
            python3.5 classify_after_cut.py ${field} 'hlt'
        ;;
        'find')
            result=$(get_pid 'mq_classify' ' '${field}' ')
            echo 'mq_classify: '${result}
        ;;
        'kill')
            result=$(kill_process)
            for line in ${result}
            do
                echo $line
            done
        ;;
        'restart')
            result=$(kill_process)
            for line in ${result}
            do
                echo $line
            done
            echo 'start process'
            path="../work_space/${field}/module/comb_hlt"
            echo ${path}
            nohup python3.5 mq_classify.py ${field} 'hlt' ${path} >/dev/null 2>nohup.out &

            echo 'done!'
        ;;
    esac
fi
