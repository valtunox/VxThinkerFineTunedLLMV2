#!/bin/bash

set -euo pipefail

echo "========================================"
echo " Kafka Messaging Service Smoke Test"
echo "========================================"
echo ""

KAFKA_HOME=${KAFKA_HOME:-$HOME/kafka_2.13-3.6.1}
BOOTSTRAP=${KAFKA_BOOTSTRAP:-localhost:9092}
LOG_FILE=${KAFKA_MESSAGING_TEST_LOG:-$HOME/kafka-messaging-test.log}
TOPICS=("messaging-events" "user-activity" "room-analytics" "chat-messages" "chat-notifications")

if [ ! -d "$KAFKA_HOME" ]; then
    echo "Kafka distribution not found at $KAFKA_HOME"
    echo "Start Kafka with kafka-all-in-one.sh before running this test."
    exit 1
fi

if [ ! -x "$KAFKA_HOME/bin/kafka-topics.sh" ]; then
    echo "Kafka CLI tools not found under $KAFKA_HOME/bin"
    exit 1
fi

touch "$LOG_FILE"

ensure_topic() {
    local topic="$1"
    if "$KAFKA_HOME/bin/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --topic "$topic" --describe >/dev/null 2>&1; then
        echo "✓ Topic '$topic' already exists"
    else
        echo "Creating topic '$topic'"
        "$KAFKA_HOME/bin/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --create --topic "$topic" --replication-factor 1 --partitions 1 >/dev/null
        echo "  ✓ Created"
    fi
}

produce_record() {
    local topic="$1"
    local key="$2"
    local payload="$3"

    printf '%s|%s\n' "$key" "$payload" | "$KAFKA_HOME/bin/kafka-console-producer.sh" \
        --bootstrap-server "$BOOTSTRAP" \
        --topic "$topic" \
        --property parse.key=true \
        --property key.separator='|' >/dev/null

    echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") $topic $key $payload" >> "$LOG_FILE"
}

consume_snapshot() {
    local topic="$1"
    echo "  Inspecting latest message..."

    if command -v timeout >/dev/null 2>&1; then
        if SNAPSHOT=$(timeout 5 "$KAFKA_HOME/bin/kafka-console-consumer.sh" \
            --bootstrap-server "$BOOTSTRAP" \
            --topic "$topic" \
            --from-beginning \
            --max-messages 1 \
            --property print.key=true \
            --property print.timestamp=true 2>/dev/null); then
            if [ -n "$SNAPSHOT" ]; then
                echo "$SNAPSHOT" | sed 's/^/    /'
            else
                echo "    (no messages consumed; topic may still be empty)"
            fi
        else
            echo "    Unable to read from topic (exit $?)"
        fi
    else
        echo "    'timeout' command missing; skipping consumer check"
    fi
}

NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
ROOM="room-demo"
USER="user-demo"

for topic in "${TOPICS[@]}"; do
    ensure_topic "$topic"

    case "$topic" in
        "messaging-events")
            payload='{"event_type":"message_sent","timestamp":"'"$NOW"'","message_data":{"room_id":"'"$ROOM"'","sender_id":"'"$USER"'","content":"Kafka messaging smoke test","message_type":"text"},"service":"messaging","version":"1.0"}'
            key="$ROOM:$USER"
            ;;
        "user-activity")
            payload='{"event_type":"user_ping","timestamp":"'"$NOW"'","user_id":"'"$USER"'","data":{"presence":"online","client":"web"},"service":"messaging"}'
            key="$USER"
            ;;
        "room-analytics")
            payload='{"event_type":"message_metrics","timestamp":"'"$NOW"'","room_id":"'"$ROOM"'","data":{"metric":"message_count","count":1},"service":"messaging"}'
            key="$ROOM"
            ;;
        "chat-messages")
            payload='{"event":"chat_message","timestamp":"'"$NOW"'","room_id":"'"$ROOM"'","message":{"id":"msg-'"$NOW"'","sender":"'"$USER"'","text":"Hello from kafka_messaging_service_test.sh"}}'
            key="chat::$ROOM"
            ;;
        "chat-notifications")
            payload='{"event":"notification","timestamp":"'"$NOW"'","target_user":"'"$USER"'","notification":{"type":"mention","room_id":"'"$ROOM"'","message":"Smoke test notification"}}'
            key="notify::$USER"
            ;;
        *)
            payload='{"timestamp":"'"$NOW"'","message":"Default payload"}'
            key="default"
            ;;
    esac

    echo "Sending sample record to '$topic'"
    produce_record "$topic" "$key" "$payload"
    consume_snapshot "$topic"
    echo ""
done

echo "Logs written to $LOG_FILE"
echo "========================================"
