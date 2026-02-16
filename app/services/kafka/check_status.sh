#!/bin/bash

echo "========================================"
echo "   Kafka & Dashboard Status Check"
echo "========================================"
echo ""

# Check if Kafka is running
echo "1. Checking Kafka process..."
if pgrep -f "kafka.Kafka" > /dev/null; then
    echo "✓ Kafka is running"
    pgrep -f "kafka.Kafka" | while read pid; do
        echo "  PID: $pid"
    done
else
    echo "✗ Kafka is NOT running"
fi

echo ""
echo "2. Checking Kafka UI process..."
if pgrep -f "kafka-ui" > /dev/null; then
    echo "✓ Kafka UI is running"
    pgrep -f "kafka-ui" | while read pid; do
        echo "  PID: $pid"
    done
else
    echo "✗ Kafka UI is NOT running"
fi

echo ""
echo "3. Checking ports..."
echo "Kafka (9092):"
if sudo lsof -i :9092 > /dev/null 2>&1; then
    sudo lsof -i :9092 | grep LISTEN
else
    echo "  Port 9092 not in use"
fi

echo ""
echo "Kafka UI (8080):"
if sudo lsof -i :8080 > /dev/null 2>&1; then
    sudo lsof -i :8080 | grep LISTEN
else
    echo "  Port 8080 not in use"
fi

echo ""
echo "4. Checking log files..."
if [ -f "$HOME/kafka.log" ]; then
    echo "Kafka log (last 10 lines):"
    tail -10 $HOME/kafka.log
else
    echo "Kafka log not found"
fi

echo ""
if [ -f "$HOME/kafka-ui.log" ]; then
    echo "Kafka UI log (last 15 lines):"
    tail -15 $HOME/kafka-ui.log
else
    echo "Kafka UI log not found"
fi

echo ""
echo "5. Testing Kafka connection..."
if [ -d "$HOME/kafka_2.13-3.6.1" ]; then
    cd $HOME/kafka_2.13-3.6.1
    timeout 5 bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092 2>&1 | head -5
else
    echo "Kafka directory not found"
fi

echo ""
echo "6. Inspecting messaging topics..."
KAFKA_HOME="$HOME/kafka_2.13-3.6.1"
MESSAGING_TOPICS=("messaging-events" "user-activity" "room-analytics" "chat-messages" "chat-notifications")

if [ -d "$KAFKA_HOME" ]; then
    for topic in "${MESSAGING_TOPICS[@]}"; do
        DESCRIBE_OUTPUT=$($KAFKA_HOME/bin/kafka-topics.sh --bootstrap-server localhost:9092 --topic "$topic" --describe 2>&1)
        if [ $? -eq 0 ] && echo "$DESCRIBE_OUTPUT" | grep -q "Topic"; then
            echo "✓ Topic '$topic'"
            echo "$DESCRIBE_OUTPUT" | sed 's/^/  /'
        else
            echo "✗ Topic '$topic' not found"
        fi
    done
else
    echo "Kafka directory not found; cannot inspect topics"
fi

echo ""
echo "========================================"
echo "Run ./kafka_messaging_service_test.sh for an end-to-end messaging smoke test."
