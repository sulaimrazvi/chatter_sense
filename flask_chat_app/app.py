from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO, emit, join_room
from sqlalchemy.orm import scoped_session, sessionmaker
from datetime import datetime
import random
import nltk

nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Initialize analyzer with custom lexicon
vader_analyzer = SentimentIntensityAnalyzer()




# Initialize VADER sentiment analyzer



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'your_secret_key'

#-----------------------



db = SQLAlchemy(app)

# ✅ FIX: Use app.app_context() to ensure we're inside the Flask app context
with app.app_context():
    SessionLocal = scoped_session(sessionmaker(bind=db.engine))

#----------------------


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

socketio = SocketIO(app, cors_allowed_origins="*")

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Friendship Model
class Friendship(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    friend_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), default="pending")  # pending | accepted

# Message Model
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Group Model
class Group(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))

# Group Member Model
class GroupMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'))

# Group Message Model
class GroupMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'))
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sentiment = db.Column(db.String(20))  # happy, sad, angry, neutral


class GroupInvite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'))
    invited_user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    invited_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    status = db.Column(db.String(20), default="pending")  # pending | accepted

class DeletedMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    message_id = db.Column(db.Integer, db.ForeignKey('message.id'))

class GroupMessageDelete(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message_id = db.Column(db.Integer, db.ForeignKey('group_message.id'), nullable=False)

class TopicMessageDelete(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message_id = db.Column(db.Integer, db.ForeignKey('topic_message.id'), nullable=False)


class Topic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TopicMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20))  # happy, sad, angry, neutral
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)



import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

vader_analyzer = SentimentIntensityAnalyzer()

from transformers import pipeline

# Load both models once
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
bart_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

label_map = {
    "joy": "happy", "happiness": "happy",
    "sadness": "sad", "fear": "sad",
    "anger": "angry", "disgust": "angry",
    "surprise": "neutral", "neutral": "neutral"
}

# Trigger words always mean angry
trigger_words = {
    "stupid": "angry", "idiot": "angry",
    "dumb": "angry", "nonsense": "angry",
    "hate": "angry", "annoying": "angry"
}

# Negation rules: phrases that negate the emotions triggered by certain words
negation_rules = {
    "not angry": "neutral", "not happy": "sad", "not sad": "neutral",
    "not stupid": "neutral", "not an idiot": "neutral", "not dumb": "neutral",
    "not annoying": "neutral", "not hate": "neutral", "no hate": "neutral",
    "i am not angry": "neutral", "i'm not angry": "neutral",
    "i am not sad": "neutral", "i'm not sad": "neutral",
    "i am not happy": "sad", "i'm not happy": "sad",
}

def get_sentiment_label(text):
    # Preprocess text
    lowered = text.lower().strip()

    # Step 1: Check negation rules first
    for phrase, label in negation_rules.items():
        if re.search(rf'\b{re.escape(phrase)}\b', lowered):  # regex for exact match
            return label  # Return neutral or sad based on negation

    # Step 2: Check trigger words for anger
    for word, sentiment in trigger_words.items():
        if re.search(rf'\b{re.escape(word)}\b', lowered):  # regex for exact match
            return sentiment  # If we detect any trigger word, return "angry"

    # Step 3: Emotion model prediction
    emotion_result = emotion_classifier(text)[0]
    emotion_top = max(emotion_result, key=lambda x: x['score'])
    emotion_label = label_map.get(emotion_top['label'].lower(), "neutral")
    emotion_score = emotion_top['score']

    # Step 4: BART model prediction
    candidate_labels = ["happy", "sad", "angry", "neutral"]
    bart_result = bart_classifier(text, candidate_labels)
    bart_label = bart_result["labels"][0]
    bart_score = bart_result["scores"][0]

    # Step 5: Decision logic based on scores
    if emotion_label == bart_label:
        return emotion_label
    elif emotion_score > 0.75:
        return emotion_label
    elif bart_score > 0.75:
        return bart_label
    else:
        return "neutral"



@login_manager.user_loader
def load_user(user_id):
    with SessionLocal() as session:
        return session.get(User, int(user_id))

# ✅ Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# ✅ Login Route
@app.route('/', methods=['GET', 'POST'])  # Allow POST for /
@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        email = request.form.get('email')  # Safely get form data
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


# ✅ Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# ✅ Dashboard Route (Shows Friends & Friend Requests)
@app.route('/dashboard')
@login_required
def dashboard():
    # Accepted friends
    friendships = Friendship.query.filter(
        ((Friendship.user_id == current_user.id) | (Friendship.friend_id == current_user.id)) &
        (Friendship.status == "accepted")
    ).all()
    
    friends = []
    for f in friendships:
        friend = User.query.get(f.friend_id if f.user_id == current_user.id else f.user_id)
        friends.append(friend)

    # Friend requests
    pending_requests = Friendship.query.filter_by(friend_id=current_user.id, status="pending").all()
    pending_users = [User.query.get(f.user_id) for f in pending_requests]

    # Other users (for sending requests)
    users = User.query.filter(User.id != current_user.id).all()

    # ✅ User's groups
    group_ids = [gm.group_id for gm in GroupMember.query.filter_by(user_id=current_user.id).all()]
    groups = Group.query.filter(Group.id.in_(group_ids)).all()

    return render_template(
        'dashboard.html',
        friends=friends,
        pending_users=pending_users,
        users=users,
        groups=groups  # ✅ pass to template
    )


# ✅ Send Friend Request
@app.route('/add_friend/<int:user_id>')
@login_required
def add_friend(user_id):
    existing_request = Friendship.query.filter_by(user_id=current_user.id, friend_id=user_id).first()
    
    if existing_request:
        flash('Friend request already sent.', 'warning')
    else:
        new_friendship = Friendship(user_id=current_user.id, friend_id=user_id, status="pending")
        db.session.add(new_friendship)
        db.session.commit()
        flash('Friend request sent!', 'success')

    return redirect(url_for('dashboard'))

# ✅ Route to list all users except the logged-in user
@app.route('/users')
@login_required
def users():
    all_users = User.query.filter(User.id != current_user.id).all()
    return render_template('users.html', users=all_users)


@app.route('/friend_requests')
@login_required
def friend_requests():
    pending_requests = Friendship.query.filter_by(friend_id=current_user.id, status="pending").all()
    pending_users = [User.query.get(f.user_id) for f in pending_requests]
    
    return render_template('friend_requests.html', pending_users=pending_users)


# ✅ Accept Friend Request
@app.route('/accept_friend/<int:user_id>')
@login_required
def accept_friend(user_id):
    friendship = Friendship.query.filter_by(user_id=user_id, friend_id=current_user.id, status="pending").first()
    
    if friendship:
        friendship.status = "accepted"
        db.session.commit()
        flash('Friend request accepted!', 'success')
    else:
        flash('Friend request not found.', 'danger')

    return redirect(url_for('dashboard'))

# ✅ Chat Route
@app.route('/chat/<int:friend_id>')
@login_required
def chat(friend_id):
    friend = User.query.get(friend_id)
    
    # Check if they are friends
    friendship = Friendship.query.filter(
        ((Friendship.user_id == current_user.id) & (Friendship.friend_id == friend_id)) |
        ((Friendship.user_id == friend_id) & (Friendship.friend_id == current_user.id))
    ).filter_by(status="accepted").first()
    
    if not friendship:
        flash("You can only chat with friends.", "danger")
        return redirect(url_for('dashboard'))

    # Get message IDs that the current user deleted
    deleted_ids = [dm.message_id for dm in DeletedMessage.query.filter_by(user_id=current_user.id).all()]

    # Fetch only visible messages (not deleted by current user)
    messages = Message.query.filter(
        (((Message.sender_id == current_user.id) & (Message.receiver_id == friend_id)) |
        ((Message.sender_id == friend_id) & (Message.receiver_id == current_user.id))) &
        (~Message.id.in_(deleted_ids))
        ).order_by(Message.timestamp).all()


    chat_room = f"chat_{min(current_user.id, friend_id)}_{max(current_user.id, friend_id)}"
    
    return render_template('chat.html', friend=friend, messages=messages, chat_room=chat_room)


@app.route('/delete_selected_messages/<int:friend_id>', methods=['POST'])
@login_required
def delete_selected_messages(friend_id):
    ids = request.form['selected_ids'].split(',')
    for message_id in ids:
        already_deleted = DeletedMessage.query.filter_by(user_id=current_user.id, message_id=message_id).first()
        if not already_deleted:
            deletion = DeletedMessage(user_id=current_user.id, message_id=message_id)
            db.session.add(deletion)
    db.session.commit()
    flash("Selected messages deleted from your view.", "info")
    return redirect(url_for('chat', friend_id=friend_id))



# Create group route
@app.route('/create_group', methods=['GET', 'POST'])
@login_required
def create_group():
    if request.method == 'POST':
        group_name = request.form['group_name']
        new_group = Group(name=group_name, created_by=current_user.id)
        db.session.add(new_group)
        db.session.commit()

        # Add current user as member
        member = GroupMember(user_id=current_user.id, group_id=new_group.id)
        db.session.add(member)
        db.session.commit()

        flash('Group created successfully!', 'success')
        return redirect(url_for('groups'))
    return render_template('create_group.html')


@app.route('/groups')
@login_required
def groups():
    
    group_ids = [gm.group_id for gm in GroupMember.query.filter_by(user_id=current_user.id).all()]
    user_groups = Group.query.filter(Group.id.in_(group_ids)).all()

    
    friendships = Friendship.query.filter(
        ((Friendship.user_id == current_user.id) | (Friendship.friend_id == current_user.id)) &
        (Friendship.status == "accepted")
    ).all()

    
    friend_ids = [
        f.friend_id if f.user_id == current_user.id else f.user_id
        for f in friendships
    ]
    friends = User.query.filter(User.id.in_(friend_ids)).all()

    return render_template('groups.html', groups=user_groups, users=friends)


@app.route('/delete_selected_group_messages/<int:group_id>', methods=['POST'])
@login_required
def delete_selected_group_messages(group_id):
    ids = request.form['selected_ids'].split(',')
    for message_id in ids:
        already_deleted = GroupMessageDelete.query.filter_by(
            user_id=current_user.id, message_id=message_id
        ).first()
        if not already_deleted:
            deletion = GroupMessageDelete(user_id=current_user.id, message_id=message_id)
            db.session.add(deletion)
    db.session.commit()
    flash("Selected messages deleted from your view.", "info")
    return redirect(url_for('group_chat', group_id=group_id))




@app.route('/remove_group_member/<int:group_id>/<int:user_id>')
@login_required
def remove_group_member(group_id, user_id):
    group = Group.query.get_or_404(group_id)

    if group.created_by != current_user.id:
        flash("Only the group creator can remove members.", "danger")
        return redirect(url_for('group_chat', group_id=group_id))

    if user_id == current_user.id:
        flash("You cannot remove yourself. Use 'Leave Group' instead.", "warning")
        return redirect(url_for('group_chat', group_id=group_id))

    membership = GroupMember.query.filter_by(group_id=group_id, user_id=user_id).first()
    if membership:
        db.session.delete(membership)
        db.session.commit()
        flash("User removed from the group.", "success")
    else:
        flash("User is not a member.", "danger")

    return redirect(url_for('group_chat', group_id=group_id))


@app.route('/leave_group/<int:group_id>')
@login_required
def leave_group(group_id):
    membership = GroupMember.query.filter_by(user_id=current_user.id, group_id=group_id).first()

    if not membership:
        flash("You are not a member of this group.", "danger")
    else:
        db.session.delete(membership)
        db.session.commit()
        flash("You left the group.", "info")

    return redirect(url_for('groups'))



@app.route('/group_chat/<int:group_id>')
@login_required
def group_chat(group_id):
    group = Group.query.get_or_404(group_id)

    # Ensure user is a member of the group
    is_member = GroupMember.query.filter_by(user_id=current_user.id, group_id=group_id).first()
    if not is_member:
        flash("You are not a member of this group.", "danger")
        return redirect(url_for('groups'))

    # Step ✅: Get IDs of messages deleted by current user in this group
    deleted_ids = db.session.query(GroupMessageDelete.message_id)\
                    .filter_by(user_id=current_user.id).all()
    deleted_ids = [id[0] for id in deleted_ids]

    # ✅ Exclude deleted messages from the result
    messages = GroupMessage.query\
        .filter_by(group_id=group_id)\
        .filter(~GroupMessage.id.in_(deleted_ids))\
        .order_by(GroupMessage.timestamp).all()

    # Map user_id to username for chat display
    user_ids = {m.sender_id for m in messages}
    user_dict = {u.id: u.username for u in User.query.filter(User.id.in_(user_ids)).all()}
    
    

    colors = ['#FF5733', '#33B5FF', '#9C33FF', '#33FF49', '#FFC733', '#FF33A8']
    user_colors = {}
    for uid in user_dict:
        user_colors[uid] = random.choice(colors)


    # Get group members for sidebar
    group_memberships = GroupMember.query.filter_by(group_id=group_id).all()
    member_ids = [gm.user_id for gm in group_memberships]
    member_users = User.query.filter(User.id.in_(member_ids)).all()

    return render_template(
    'group_chat.html',
    group=group,
    messages=messages,
    user_dict=user_dict,
    group_members=member_users,
    creator_id=group.created_by,
    current_user_id=current_user.id,
    user_colors=user_colors
)




@app.route('/send_group_invite/<int:user_id>/<int:group_id>')
@login_required
def send_group_invite(user_id, group_id):
    group = Group.query.get_or_404(group_id)

    if group.created_by != current_user.id:
        flash("Only the group creator can send invites.", "danger")
        return redirect(url_for('groups'))

    already_member = GroupMember.query.filter_by(user_id=user_id, group_id=group_id).first()
    already_invited = GroupInvite.query.filter_by(invited_user_id=user_id, group_id=group_id, status="pending").first()

    if already_member:
        flash("User is already a member.", "info")
    elif already_invited:
        flash("Invitation already sent.", "info")
    else:
        invite = GroupInvite(group_id=group_id, invited_user_id=user_id, invited_by=current_user.id)
        db.session.add(invite)
        db.session.commit()
        flash("Invitation sent!", "success")

    return redirect(url_for('groups'))


@app.route('/accept_group_invite/<int:invite_id>')
@login_required
def accept_group_invite(invite_id):
    invite = GroupInvite.query.get_or_404(invite_id)

    if invite.invited_user_id != current_user.id:
        flash("You are not authorized for this invite.", "danger")
        return redirect(url_for('dashboard'))

    invite.status = "accepted"
    member = GroupMember(user_id=current_user.id, group_id=invite.group_id)
    db.session.add(member)
    db.session.commit()
    flash("You joined the group!", "success")
    return redirect(url_for('groups'))


@app.route('/group_invites')
@login_required
def group_invites():
    invites = GroupInvite.query.filter_by(invited_user_id=current_user.id, status="pending").all()

    # Create mapping dictionaries
    group_ids = {invite.group_id for invite in invites}
    inviter_ids = {invite.invited_by for invite in invites}

    group_map = {g.id: g.name for g in Group.query.filter(Group.id.in_(group_ids)).all()}
    user_map = {u.id: u.username for u in User.query.filter(User.id.in_(inviter_ids)).all()}

    return render_template(
        'group_invites.html',
        invites=invites,
        group_map=group_map,
        user_map=user_map
    )
#---------------------------------------------------------------------------------------------------

@app.route('/create_topic', methods=['GET', 'POST'])
@login_required
def create_topic():
    if request.method == 'POST':
        title = request.form['title']
        new_topic = Topic(title=title, created_by=current_user.id)
        db.session.add(new_topic)
        db.session.commit()
        flash("Topic created!", "success")
        return redirect(url_for('topics'))
    return render_template('create_topic.html')


@app.route('/topics')
@login_required
def topics():
    all_topics = Topic.query.order_by(Topic.created_at.desc()).all()
    return render_template('topics.html', topics=all_topics)


@app.route('/topic_chat/<int:topic_id>')
@login_required
def topic_chat(topic_id):
    topic = Topic.query.get_or_404(topic_id)

    # Exclude deleted messages for this user
    deleted_ids = db.session.query(TopicMessageDelete.message_id).filter_by(user_id=current_user.id).all()
    deleted_ids = [id[0] for id in deleted_ids]

    messages = TopicMessage.query.filter_by(topic_id=topic_id)\
        .filter(~TopicMessage.id.in_(deleted_ids))\
        .order_by(TopicMessage.timestamp).all()

    user_ids = {m.sender_id for m in messages}
    user_dict = {u.id: u.username for u in User.query.filter(User.id.in_(user_ids)).all()}
    colors = ['#FF5733', '#33B5FF', '#9C33FF', '#33FF49', '#FFC733', '#FF33A8']
    user_colors = {uid: random.choice(colors) for uid in user_dict}

    return render_template(
        'topic_chat.html',
        topic=topic,
        messages=messages,
        user_dict=user_dict,
        current_user_id=current_user.id,
        user_colors=user_colors
    )


@app.route('/delete_selected_topic_messages/<int:topic_id>', methods=['POST'])
@login_required
def delete_selected_topic_messages(topic_id):
    ids = request.form['selected_ids'].split(',')
    for message_id in ids:
        if not TopicMessageDelete.query.filter_by(user_id=current_user.id, message_id=message_id).first():
            db.session.add(TopicMessageDelete(user_id=current_user.id, message_id=message_id))
    db.session.commit()
    flash("Selected messages deleted from your view.", "info")
    return redirect(url_for('topic_chat', topic_id=topic_id))

#-------------------------------------------------------------------------------------------------------
@socketio.on('join_chat')
def join_chat(data):
    """ ✅ Ensure users join the correct private chat room """
    friend_id = int(data['friend_id'])  # Convert to integer
    chat_room = f"chat_{min(current_user.id, friend_id)}_{max(current_user.id, friend_id)}"

    join_room(chat_room)
    print(f"{current_user.username} joined room {chat_room}")


@socketio.on('send_message')
def handle_message(data):
    receiver_id = int(data['receiver_id'])
    content = data['content']

    sentiment = get_sentiment_label(content)  # Use the BART classifier

    # Save message
    message = Message(sender_id=current_user.id, receiver_id=receiver_id, content=content)
    db.session.add(message)
    db.session.commit()

    chat_room = f"chat_{min(current_user.id, receiver_id)}_{max(current_user.id, receiver_id)}"

    emit('receive_message', {
        'sender': current_user.username,
        'content': content,
        'sentiment': sentiment
    }, room=chat_room)




@socketio.on('join_group')
def join_group(data):
    group_id = data['group_id']
    join_room(f"group_{group_id}")
    print(f"{current_user.username} joined group {group_id}")

@socketio.on('send_group_message')
def handle_group_message(data):
    group_id = data['group_id']
    content = data['content']

    sentiment = get_sentiment_label(content)  # ✅ Use the custom logic

    message = GroupMessage(
        group_id=group_id,
        sender_id=current_user.id,
        content=content,
        sentiment=sentiment
    )
    db.session.add(message)
    db.session.commit()

    emit('receive_group_message', {
        'sender': current_user.username,
        'content': content,
        'sentiment': sentiment,
        'color': "#000"  # optional: keep for frontend
    }, room=f"group_{group_id}")


@socketio.on('join_topic')
def join_topic(data):
    topic_id = data['topic_id']
    join_room(f"topic_{topic_id}")
    print(f"{current_user.username} joined topic {topic_id}")

@socketio.on('send_topic_message')
def handle_topic_message(data):
    topic_id = data['topic_id']
    content = data['content']
    sentiment = get_sentiment_label(content)

    message = TopicMessage(topic_id=topic_id, sender_id=current_user.id, content=content, sentiment=sentiment)
    db.session.add(message)
    db.session.commit()

    emit('receive_topic_message', {
        'sender': current_user.username,
        'content': content,
        'sentiment': sentiment,
        'color': "#000"  # can be randomized later like group
    }, room=f"topic_{topic_id}")






if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)
